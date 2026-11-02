import io
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile

from bankstreet.fraud.inference import score_transactions


PROJECT_FOLDER = Path(__file__).resolve().parent
MAX_FILE_BYTES = 50 * 1024 * 1024

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BankStreet AI — Fraud API",
    version="1.0.0",
    description="Score compatible ULB-format transaction CSV files.",
)


@app.get("/health")
def health():
    # This checks that the API responds, not model performance.
    return {"status": "ok", "service": "BankStreet Fraud API"}


@app.post("/score-csv")
def score_csv(file: Annotated[UploadFile, File()]):
    try:
        content = file.file.read(MAX_FILE_BYTES + 1)

        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="CSV must not exceed 50 MiB.",
            )

        if not content.strip():
            raise HTTPException(
                status_code=422,
                detail="The uploaded file is empty.",
            )

        try:
            transactions = pd.read_csv(io.BytesIO(content))
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(
                status_code=422,
                detail="Could not read the CSV file.",
            ) from error

        scored, threshold = score_transactions(
            transactions,
            PROJECT_FOLDER,
        )

        return {
            "transactions_scored": len(scored),
            "flagged_for_review": int(
                scored["review_required"].sum()
            ),
            "threshold": float(threshold),
            "results": scored.to_dict(orient="records"),
        }

    except HTTPException:
        raise
    except ValueError as error:
        if str(error) == "Model does not match the frozen policy.":
            logger.exception("Model-policy mismatch")
            raise HTTPException(
                status_code=500,
                detail="Model configuration error.",
            ) from error

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except (OSError, KeyError, xgb.core.XGBoostError) as error:
        logger.exception("Fraud scoring failed")
        raise HTTPException(
            status_code=500,
            detail="Scoring service is unavailable.",
        ) from error
    finally:
        file.file.close()