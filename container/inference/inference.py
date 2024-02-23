from io import StringIO
import flask
from flask import Flask, Response, Request
import logging

from bionemo.triton.inference_wrapper import new_inference_wrapper
import warnings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

app = Flask(__name__)
connection = new_inference_wrapper("grpc://localhost:8001")


@app.route("/ping", methods=["GET"])
def ping():
    """
    Check the health of the model server by verifying if the model is loaded.

    Returns a 200 status code if the model is loaded successfully, or a 500
    status code if there is an error.

    Returns:
        flask.Response: A response object containing the status code and mimetype.
    """
    status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Handle prediction requests by preprocessing the input data, making predictions,
    and returning the predictions as a JSON object.

    This function checks if the request content type is supported (text/csv),
    and if so, decodes the input data, preprocesses it, makes predictions, and returns
    the predictions as a JSON object. If the content type is not supported, a 415 status
    code is returned.

    Returns:
        flask.Response: A response object containing the predictions, status code, and mimetype.
    """
    print(f"Predictor: received content type: {flask.request.content_type}")
    if flask.request.content_type == "text/csv":
        input = flask.request.data.decode("utf-8")
        print(f"Predictor: received input: {input}")
        seqs = input.split(",")
        embeddings = connection.seqs_to_embedding(seqs)
        print(f"{embeddings.shape=}")
        print(f"Predictor: output: {embeddings}")
        # Return the predictions as a list
        return embeddings.tolist()
    else:
        print(f"Received: {flask.request.content_type}", flush=True)
        return flask.Response(
            response=f"This predictor only supports CSV data; Received: {flask.request.content_type}",
            status=415,
            mimetype="text/plain",
        )
