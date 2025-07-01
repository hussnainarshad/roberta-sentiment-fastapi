# RoBERTa Sentiment Analysis API 🚀

This repository provides a REST API for sentiment analysis using the `cardiffnlp/twitter-roberta-base-sentiment` model. It uses **FastAPI** for the backend and **Transformers** for model inference.

## 📦 Features

- Sentiment classification: `positive`, `neutral`, or `negative`
- Confidence score for each prediction
- Interactive Swagger UI (`/docs`)
- Powered by HuggingFace Transformers and RoBERTa

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
