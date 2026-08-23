
# Core libraries
import sys
import logging

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# App & logging setup
# ---------------------------------------------------------------------------
superkart_api = Flask("superkart_sales_app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("Starting SuperKart Sales API (module=%s, root=%s)", __name__, superkart_api.root_path)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = "forecast_superkart_sales_model.joblib"
REQUIRED_FIELDS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age_Years",
    "Product_Type_Category",
    "Product_Id_char",
]

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model from %s", MODEL_PATH)
    raise


def build_record(payload: dict) -> dict:
    """Pull only the expected fields out of an incoming payload.

    Raises KeyError (caught by the caller) if a required field is absent.
    """
    return {field: payload[field] for field in REQUIRED_FIELDS}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@superkart_api.route("/", methods=["GET"])
def home():
    """Landing page with basic usage info."""
    logger.info("Home endpoint accessed")

    html = """
      <!DOCTYPE html>
      <html>
      <head>
        <title>SuperKart Sales API</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f4f4f4; }
          h1 { color: #333; font-size: 2.5em; }
          p { color: #666; font-size: 1.2em; margin-top: 15px; }
          code { background: #eee; padding: 2px 6px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>SuperKart Sales Prediction API</h1>
        <p>POST a single product's features to <code>/v1/predict</code> for one prediction.</p>
        <p>POST a CSV file to <code>/v1/predict/batch</code> for predictions on multiple products.</p>
      </body>
      </html>
    """
    return html


@superkart_api.route("/health", methods=["GET"])
def health():
    """Simple health check for uptime monitoring / load balancers."""
    return jsonify({"status": "ok"})


@superkart_api.route("/v1/predict", methods=["POST"])
def predict_sales():
    """Predict sales for a single product from a JSON payload."""
    try:
        payload = request.get_json(force=True, silent=False)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        record = build_record(payload)
        input_data = pd.DataFrame([record])
        logger.debug("Single prediction input:\n%s", input_data)

        prediction = float(model.predict(input_data)[0])
        return jsonify({"Sales": round(prediction, 2)})

    except KeyError as e:
        logger.warning("Missing field in request: %s", e)
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@superkart_api.route("/v1/predict/batch", methods=["POST"])
def predict_sales_batch():
    """Predict sales for multiple products uploaded as a CSV file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Attach a CSV under the 'file' key."}), 400

        uploaded_file = request.files["file"]
        input_data = pd.read_csv(uploaded_file)

        missing_cols = [col for col in REQUIRED_FIELDS if col not in input_data.columns]
        if missing_cols:
            return jsonify({"error": f"CSV is missing required columns: {missing_cols}"}), 400

        logger.debug("Batch prediction input shape: %s", input_data.shape)

        predictions = model.predict(input_data[REQUIRED_FIELDS]).tolist()
        results = [round(float(p), 2) for p in predictions]

        return jsonify({"Sales": results, "count": len(results)})

    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded CSV is empty."}), 400
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


if __name__ == "__main__":
    superkart_api.run(debug=True)
