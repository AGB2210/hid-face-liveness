@rem Copy this file to aws-credentials.bat (git-ignored) and fill it in.
@rem Values: Workshop Studio event page -> "Get AWS CLI credentials" -> Windows tab.
@rem They expire after a few hours - paste fresh ones when Rekognition stops working.
set AWS_DEFAULT_REGION=us-east-1
set AWS_ACCESS_KEY_ID=PASTE_HERE
set AWS_SECRET_ACCESS_KEY=PASTE_HERE
set AWS_SESSION_TOKEN=PASTE_HERE
@rem Use Rekognition for face matching (remove this line to stay on local ArcFace):
set HID_IDENTITY=rekognition
