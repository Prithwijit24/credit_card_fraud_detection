from __future__ import annotations

import json
import os
from typing import Any


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Streamlit fraud detection console.")
    parser.parse_args()

    try:
        import requests
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install UI dependencies with `pip install -e .[ui]`.") from exc

    api_url = os.getenv("FRAUD_API_URL", "http://localhost:8000")
    st.set_page_config(page_title="Fraud Detection Console", layout="wide")
    st.title("Credit Card Fraud Detection Console")

    with st.sidebar:
        st.subheader("Backend")
        api_url = st.text_input("FastAPI URL", api_url)
        if st.button("Health check"):
            response = requests.get(f"{api_url}/health", timeout=5)
            st.json(response.json())

    tab_score, tab_metadata = st.tabs(["Score transaction", "Model metadata"])
    with tab_score:
        default_payload: dict[str, Any] = {
            "accountNumber": "10001",
            "customerId": "cust-1",
            "creditLimit": 5000,
            "availableMoney": 4200,
            "transactionDateTime": "2016-08-01T23:45:00",
            "transactionAmount": 125.5,
            "merchantName": "merchant_a",
            "acqCountry": "US",
            "merchantCountryCode": "US",
            "posEntryMode": "05",
            "posConditionCode": "01",
            "merchantCategoryCode": "online_retail",
            "currentExpDate": "2020-12-01",
            "accountOpenDate": "2015-01-01",
            "dateOfLastAddressChange": "2016-01-01",
            "cardCVV": "123",
            "enteredCVV": "123",
            "cardLast4Digits": "1111",
            "transactionType": "PURCHASE",
            "currentBalance": 800,
            "merchantCity": "Austin",
            "merchantState": "TX",
            "merchantZip": "78701",
            "cardPresent": False,
            "expirationDateKeyInMatch": True,
            "isFraud": False,
        }
        payload_text = st.text_area(
            "Transaction JSON",
            json.dumps(default_payload, indent=2),
            height=420,
        )
        if st.button("Score"):
            payload = json.loads(payload_text)
            response = requests.post(f"{api_url}/score", json=payload, timeout=10)
            st.json(response.json())

    with tab_metadata:
        if st.button("Load metadata"):
            response = requests.get(f"{api_url}/metadata", timeout=10)
            st.json(response.json())


if __name__ == "__main__":
    main()
