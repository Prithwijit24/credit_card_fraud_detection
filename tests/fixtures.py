from __future__ import annotations

from typing import Any


def notebook_transaction(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "accountNumber": "10001",
        "customerId": "cust-1",
        "creditLimit": 5000.0,
        "availableMoney": 4200.0,
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
        "echoBuffer": "",
        "currentBalance": 800.0,
        "merchantCity": "Austin",
        "merchantState": "TX",
        "merchantZip": "78701",
        "cardPresent": False,
        "posOnPremises": "",
        "recurringAuthInd": "",
        "expirationDateKeyInMatch": True,
        "isFraud": False,
    }
    row.update(overrides)
    return row
