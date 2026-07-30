import os
import aws_cdk as cdk
from stack.gym_ml_stack import GymMlStack

app = cdk.App()

GymMlStack(
    app,
    "GymMlStack",
    existing_table_name="AssistantData",
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ["CDK_DEFAULT_REGION"],
    ),
)

cdk.Tags.of(app).add("Project","calendar-agent")
cdk.Tags.of(app).add("Feature", "gym-ml")
cdk.Tags.of(app).add("Environment", "dev")

app.synth()
