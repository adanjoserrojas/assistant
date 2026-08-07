from ml import fetch_sessions, build_training_data


def lambda_handler(event, context):
    sessions = fetch_sessions()
    if not sessions:
        raise RuntimeError("There is no data in the DynamoDB table, user needs to log")

    training_data = build_training_data(sessions)
    return training_data
