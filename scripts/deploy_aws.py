"""Deploy the kiosk to AWS: EC2 (app) + CloudFront (HTTPS) + S3 (code) + Rekognition.

Uses the credentials from aws-credentials.bat (run via deploy.bat) or any AWS profile.

  python scripts/deploy_aws.py            deploy (or show the existing deployment)
  python scripts/deploy_aws.py status     health of the running deployment
  python scripts/deploy_aws.py update     ship the current code and restart (keeps data)
  python scripts/deploy_aws.py destroy    remove everything this script created

State is kept in data/aws_deploy.json (git-ignored).
"""
from __future__ import annotations

import io
import json
import secrets
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "aws_deploy.json"
REGION = "us-east-1"
NAME = "hid-kiosk"
PORT = 8000
INSTANCE_TYPES = ["c6i.xlarge", "c5.xlarge", "t3.xlarge", "t3.large"]
UBUNTU_AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
CF_CACHING_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"        # AWS managed cache policy
CF_ALL_VIEWER_EXCEPT_HOST = "b689b0a8-53d0-40ab-baf2-68738e2966ac"  # AWS managed origin request policy

session = boto3.Session(region_name=REGION)
ec2, s3, iam, cf, ssm, sts = (session.client(n) for n in ("ec2", "s3", "iam", "cloudfront", "ssm", "sts"))


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def code(e: ClientError) -> str:
    return e.response["Error"]["Code"]


# --------------------------------------------------------------------------------- code
def package_code() -> bytes:
    """Tracked + untracked-but-not-ignored files (no models, data, secrets, venv)."""
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            p = ROOT / f
            if f and p.is_file() and not f.endswith(".bat"):
                tar.add(p, arcname=f)
    return buf.getvalue()


def upload(st: dict) -> dict:
    """Upload code + the two small exported MiniFASNet ONNX files; return presigned URLs."""
    bucket = st["bucket"]
    s3.put_object(Bucket=bucket, Key="code.tar.gz", Body=package_code())
    keys = {"code": "code.tar.gz"}
    for name in ("minifas_v2_2.7.onnx", "minifas_v1se_4.0.onnx"):
        s3.upload_file(str(ROOT / "models" / name), bucket, f"models/{name}")
        keys[name] = f"models/{name}"
    return {k: s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": v}, ExpiresIn=6 * 3600)
            for k, v in keys.items()}


# ---------------------------------------------------------------------------- user data
def user_data(urls: dict, identity: str, password: str) -> str:
    return f"""#!/bin/bash
set -euxo pipefail
log() {{ echo "HIDSETUP: $*" | tee /dev/console; }}
log "starting setup"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 libegl1 libgles2 curl
mkdir -p /opt/hid && cd /opt/hid
curl -fsSL '{urls["code"]}' -o /tmp/code.tar.gz && tar xzf /tmp/code.tar.gz -C /opt/hid
log "code unpacked"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
log "packages installed"
.venv/bin/python scripts/get_models.py
curl -fsSL '{urls["minifas_v2_2.7.onnx"]}' -o models/minifas_v2_2.7.onnx
curl -fsSL '{urls["minifas_v1se_4.0.onnx"]}' -o models/minifas_v1se_4.0.onnx
log "models ready"
cat > /etc/hid-kiosk.env <<'ENV'
HID_PROFILE=cloud
HID_IDENTITY={identity}
HID_PASSWORD={password}
AWS_DEFAULT_REGION={REGION}
GLOG_minloglevel=2
ENV
chmod 600 /etc/hid-kiosk.env
cat > /etc/systemd/system/hid-kiosk.service <<'UNIT'
[Unit]
Description=HID liveness kiosk
After=network-online.target
[Service]
WorkingDirectory=/opt/hid
EnvironmentFile=/etc/hid-kiosk.env
ExecStart=/opt/hid/.venv/bin/python -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port {PORT}
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now hid-kiosk
log "service started"
"""


UPDATE_SCRIPT = """set -e
cd /opt/hid
curl -fsSL '{url}' -o /tmp/code.tar.gz
tar xzf /tmp/code.tar.gz -C /opt/hid
.venv/bin/pip install -q -r requirements.txt
systemctl restart hid-kiosk
echo updated
"""


# ------------------------------------------------------------------------------ pieces
def ensure_bucket(st: dict, account: str) -> None:
    if st.get("bucket"):
        return
    bucket = f"{NAME}-{account}-{REGION}"
    try:
        s3.create_bucket(Bucket=bucket)
    except ClientError as e:
        if code(e) != "BucketAlreadyOwnedByYou":
            raise
    st["bucket"] = bucket
    log(f"S3 bucket {bucket}")


def ensure_role(st: dict) -> None:
    """Instance role for Rekognition (+ SSM for in-place updates). Optional."""
    if "role_ok" in st:
        return
    role = f"{NAME}-ec2-role"
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        try:
            iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust))
        except ClientError as e:
            if code(e) != "EntityAlreadyExists":
                raise
        for arn in ("arn:aws:iam::aws:policy/AmazonRekognitionFullAccess",
                    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"):
            iam.attach_role_policy(RoleName=role, PolicyArn=arn)
        try:
            iam.create_instance_profile(InstanceProfileName=role)
            iam.add_role_to_instance_profile(InstanceProfileName=role, RoleName=role)
        except ClientError as e:
            if code(e) not in ("EntityAlreadyExists", "LimitExceeded"):
                raise
        st.update(role_ok=True, role=role)
        log(f"IAM role {role} (Rekognition + SSM)")
        time.sleep(10)  # IAM propagation
    except ClientError as e:
        st.update(role_ok=False)
        log(f"IAM role not allowed here ({code(e)}) - server will use the local face model instead of Rekognition")


def my_ip() -> str:
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def ensure_sg(st: dict) -> None:
    if st.get("sg"):
        return
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        sys.exit("No default VPC in this account - cannot continue automatically.")
    vpc = vpcs[0]["VpcId"]
    try:
        sg = ec2.create_security_group(GroupName=f"{NAME}-sg", Description="HID kiosk app port", VpcId=vpc)["GroupId"]
    except ClientError as e:
        if code(e) != "InvalidGroup.Duplicate":
            raise
        sg = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"{NAME}-sg"]}])["SecurityGroups"][0]["GroupId"]
    perms = [{"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT,
              "IpRanges": [{"CidrIp": f"{my_ip()}/32", "Description": "deployer laptop"}]}]
    try:
        pl = ec2.describe_managed_prefix_lists(
            Filters=[{"Name": "prefix-list-name", "Values": ["com.amazonaws.global.cloudfront.origin-facing"]}])["PrefixLists"]
    except ClientError:
        pl = []
    if pl:  # only CloudFront may reach the app port
        perms.append({"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT,
                      "PrefixListIds": [{"PrefixListId": pl[0]["PrefixListId"], "Description": "CloudFront"}]})
    else:
        perms.append({"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]})
    try:
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=perms)
    except ClientError as e:
        if code(e) != "InvalidPermission.Duplicate":
            raise
    st.update(sg=sg, vpc=vpc)
    log(f"security group {sg} (port {PORT}: CloudFront + your IP only)")


def launch(st: dict, urls: dict) -> str:
    ami = ssm.get_parameter(Name=UBUNTU_AMI_PARAM)["Parameter"]["Value"]
    identity = "rekognition" if st.get("role_ok") else "local_arcface"
    ud = user_data(urls, identity, st["password"])
    last = None
    for itype in INSTANCE_TYPES:
        for attempt in range(4):
            args = dict(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1, UserData=ud,
                        SecurityGroupIds=[st["sg"]],
                        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                              "Ebs": {"VolumeSize": 30, "VolumeType": "gp3"}}],
                        MetadataOptions={"HttpTokens": "required"},
                        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}]}])
            if st.get("role_ok"):
                args["IamInstanceProfile"] = {"Name": st["role"]}
            try:
                inst = ec2.run_instances(**args)["Instances"][0]["InstanceId"]
                st.update(instance=inst, instance_type=itype, identity=identity)
                log(f"launched {inst} ({itype}, identity={identity})")
                return inst
            except ClientError as e:
                last = e
                msg = str(e)
                if "instance profile" in msg.lower() and attempt < 3:
                    time.sleep(8)  # IAM still propagating
                    continue
                if code(e) in ("UnauthorizedOperation", "InvalidParameterValue", "Unsupported",
                               "InsufficientInstanceCapacity", "VcpuLimitExceeded") or "not authorized" in msg:
                    log(f"{itype} not available ({code(e)}), trying next type")
                    break
                raise
    raise SystemExit(f"Could not launch any instance type: {last}")


def wait_running(inst: str) -> str:
    ec2.get_waiter("instance_running").wait(InstanceIds=[inst])
    dns = ec2.describe_instances(InstanceIds=[inst])["Reservations"][0]["Instances"][0]["PublicDnsName"]
    log(f"instance running: {dns}")
    return dns


def ensure_cloudfront(st: dict) -> None:
    if st.get("cf_id"):
        return
    ref = f"{NAME}-{int(time.time())}"
    cfg = {
        "CallerReference": ref, "Comment": "HID liveness kiosk (HTTPS for camera access)", "Enabled": True,
        "Origins": {"Quantity": 1, "Items": [{
            "Id": "ec2", "DomainName": st["dns"],
            "CustomOriginConfig": {"HTTPPort": PORT, "HTTPSPort": 443, "OriginProtocolPolicy": "http-only",
                                   "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 5}}]},
        "DefaultCacheBehavior": {
            "TargetOriginId": "ec2", "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": {"Quantity": 7, "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                               "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
            "CachePolicyId": CF_CACHING_DISABLED, "OriginRequestPolicyId": CF_ALL_VIEWER_EXCEPT_HOST,
            "Compress": True},
        "PriceClass": "PriceClass_All",
    }
    d = cf.create_distribution(DistributionConfig=cfg)["Distribution"]
    st.update(cf_id=d["Id"], url=f"https://{d['DomainName']}")
    log(f"CloudFront {d['Id']} -> {st['url']} (takes ~5-10 min to go live)")


def http_ok(url: str, password: str, timeout: float = 5) -> bool:
    import hashlib
    token = hashlib.sha256(f"hid:{password}".encode()).hexdigest()  # same cookie the login page sets
    req = urllib.request.Request(url, headers={"Cookie": f"hid_auth={token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def console_progress(inst: str) -> str:
    try:
        out = ec2.get_console_output(InstanceId=inst, Latest=True).get("Output", "")
    except ClientError:
        return ""
    lines = [l.split("HIDSETUP:", 1)[1].strip() for l in out.splitlines() if "HIDSETUP:" in l]
    return lines[-1] if lines else ""


def wait_app(st: dict, minutes: int = 25) -> bool:
    direct = f"http://{st['dns']}:{PORT}/api/config"
    last = ""
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        if http_ok(direct, st["password"]):
            log(f"app is up on the server ({int(time.time() - t0)} s)")
            return True
        prog = console_progress(st["instance"])
        if prog and prog != last:
            log(f"server setup: {prog}")
            last = prog
        time.sleep(15)
    log("app did not come up in time - check EC2 console > instance > Actions > Monitor > Get system log")
    return False


def wait_cloudfront(st: dict, minutes: int = 20) -> bool:
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        if http_ok(st["url"] + "/api/config", st["password"], timeout=10):
            return True
        time.sleep(20)
    return False


# ------------------------------------------------------------------------------ commands
def deploy() -> None:
    st = load_state()
    ident = sts.get_caller_identity()
    log(f"account {ident['Account']} as {ident['Arn'].split('/')[-2] if '/' in ident['Arn'] else ident['Arn']}")
    if st.get("instance") and st.get("url"):
        log(f"already deployed: {st['url']}  (use 'update' to ship new code, 'destroy' to remove)")
        return status()
    st.setdefault("password", secrets.token_urlsafe(9))
    save_state(st)
    ensure_bucket(st, ident["Account"]); save_state(st)
    ensure_role(st); save_state(st)
    ensure_sg(st); save_state(st)
    urls = upload(st)
    log("code uploaded to S3")
    launch(st, urls); save_state(st)
    st["dns"] = wait_running(st["instance"]); save_state(st)
    ensure_cloudfront(st); save_state(st)
    log("server is installing itself (packages + models), usually 5-10 min...")
    wait_app(st)
    log("waiting for CloudFront to go live...")
    live = wait_cloudfront(st)
    print()
    print("=" * 60)
    print(f"  Kiosk URL : {st['url']}" + ("" if live else "   (CloudFront still deploying - try in a few minutes)"))
    print(f"  Password  : {st['password']}")
    print(f"  Identity  : {st.get('identity')}")
    print("=" * 60)


def status() -> None:
    st = load_state()
    if not st.get("instance"):
        log("nothing deployed yet")
        return
    inst = ec2.describe_instances(InstanceIds=[st["instance"]])["Reservations"][0]["Instances"][0]
    log(f"instance {st['instance']} {inst['State']['Name']} ({st.get('instance_type')}), identity={st.get('identity')}")
    if st.get("dns"):
        up = http_ok(f"http://{st['dns']}:{PORT}/api/config", st["password"])
        log(f"server direct : {'UP' if up else 'DOWN'}  (last setup step: {console_progress(st['instance']) or '-'})")
    if st.get("url"):
        log(f"public HTTPS  : {'UP' if http_ok(st['url'] + '/api/config', st['password'], 10) else 'not yet'}  {st['url']}")
        log(f"password      : {st['password']}")


def update() -> None:
    st = load_state()
    if not st.get("instance"):
        sys.exit("Nothing deployed yet - run without arguments first.")
    urls = upload(st)
    log("new code uploaded")
    if st.get("role_ok"):
        try:
            cmd = ssm.send_command(InstanceIds=[st["instance"]], DocumentName="AWS-RunShellScript",
                                   Parameters={"commands": [UPDATE_SCRIPT.format(url=urls["code"])]})["Command"]["CommandId"]
            for _ in range(60):
                time.sleep(5)
                try:
                    r = ssm.get_command_invocation(CommandId=cmd, InstanceId=st["instance"])
                except ClientError:
                    continue
                if r["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                    log(f"in-place update: {r['Status']} {r.get('StandardErrorContent', '')[-300:]}")
                    break
            if wait_app(st, 5):
                return
        except ClientError as e:
            log(f"in-place update not possible ({code(e)}) - relaunching the server instead")
    # Fallback: new instance, repoint CloudFront, terminate the old one (enrolled users are lost)
    old = st["instance"]
    launch(st, urls)
    st["dns"] = wait_running(st["instance"]); save_state(st)
    wait_app(st)
    c = cf.get_distribution_config(Id=st["cf_id"])
    c["DistributionConfig"]["Origins"]["Items"][0]["DomainName"] = st["dns"]
    cf.update_distribution(Id=st["cf_id"], IfMatch=c["ETag"], DistributionConfig=c["DistributionConfig"])
    ec2.terminate_instances(InstanceIds=[old])
    save_state(st)
    log("relaunched; CloudFront is switching to the new server (a few minutes). Re-enroll users.")


def destroy() -> None:
    st = load_state()
    if st.get("cf_id"):
        c = cf.get_distribution_config(Id=st["cf_id"])
        if c["DistributionConfig"]["Enabled"]:
            c["DistributionConfig"]["Enabled"] = False
            cf.update_distribution(Id=st["cf_id"], IfMatch=c["ETag"], DistributionConfig=c["DistributionConfig"])
            log("CloudFront disabled - run 'destroy' again in ~15 min to delete it")
        else:
            try:
                cf.delete_distribution(Id=st["cf_id"], IfMatch=c["ETag"])
                st.pop("cf_id"); st.pop("url", None)
                log("CloudFront deleted")
            except ClientError as e:
                log(f"CloudFront not ready to delete yet ({code(e)}) - try again later")
    if st.get("instance"):
        ec2.terminate_instances(InstanceIds=[st["instance"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[st["instance"]])
        st.pop("instance"); st.pop("dns", None)
        log("instance terminated")
    if st.get("sg"):
        try:
            ec2.delete_security_group(GroupId=st["sg"]); st.pop("sg")
            log("security group deleted")
        except ClientError as e:
            log(f"security group kept ({code(e)})")
    if st.get("bucket"):
        b = session.resource("s3").Bucket(st["bucket"])
        b.objects.all().delete(); b.delete(); st.pop("bucket")
        log("S3 bucket deleted")
    save_state(st)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    {"deploy": deploy, "status": status, "update": update, "destroy": destroy}[cmd]()
