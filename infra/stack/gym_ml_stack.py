from aws_cdk import (

    Stack, 
    RemovalPolicy, 
    CfnOutput,

)
from constructs import Construct
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as lambd




class GymMlStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        existing_table_name: str,
        **kwargs,
    ) -> None:
        
        super().__init__(scope, construct_id, **kwargs)

        self.existing_table_name = existing_table_name
        self.artifacts_bucket = s3.Bucket(
            self, "MlArtifacts",
            bucket_name=f"gym-ml-artifacts-{self.account}-{self.region}",
            versioned=True,                                      # history + rollback
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,                 # cdk destroy won't eat your data
        )
        self.artifacts_function = lambd.function( 
        )
        

        # calendar-agent-dev is an IAM user, not a role -- from_role_name would
        # emit a policy CloudFormation cannot attach (404 NotFound at deploy).
        agent_user = iam.User.from_user_name(self, "CalendarAgentDev", "calendar-agent-dev")
        self.artifacts_bucket.grant_read_write(agent_user)

        CfnOutput(self, "ArtifactsBucketName", value=self.artifacts_bucket.bucket_name)
