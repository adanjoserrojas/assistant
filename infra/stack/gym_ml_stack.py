from aws_cdk import Stack
from constructs import Construct


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