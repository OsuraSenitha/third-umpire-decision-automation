def handler(event, context):
    print(event)
    annotations = [
        [10,20,30,40],
        [50,60,70,80]
    ]
    print(f"Sending annotations: {annotations}")
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "annotations ": annotations
        }
    }