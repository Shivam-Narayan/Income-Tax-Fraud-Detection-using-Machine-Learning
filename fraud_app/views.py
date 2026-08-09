import json
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.serializers import BooleanField, CharField, FloatField, IntegerField, JSONField, Serializer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'fraud_app' / 'model.pkl'
METADATA_PATH = BASE_DIR / 'artifacts' / 'model_metadata.json'

POSITIVE_LABEL = '>50K'
DEFAULT_THRESHOLD = 0.5


def _load_metadata():
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding='utf-8'))


def _build_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


MODEL = _build_model()
METADATA = _load_metadata()
DECISION_THRESHOLD = float(METADATA.get('decision_threshold', DEFAULT_THRESHOLD))


def _format_prediction(label, probability):
    is_fraud = label == POSITIVE_LABEL or probability >= DECISION_THRESHOLD
    return 'Likely fraud' if is_fraud else 'Likely safe'


class PredictionInputSerializer(Serializer):
    age = IntegerField()
    workclass = CharField()
    education_num = IntegerField()
    marital_status = CharField()
    occupation = CharField()
    relationship = CharField()
    race = CharField()
    sex = CharField()
    capital_gain = IntegerField()
    capital_loss = IntegerField()
    hours_per_week = IntegerField()
    native_country = CharField()


class PredictionResponseSerializer(Serializer):
    prediction = CharField()
    probability = FloatField()
    threshold = FloatField()
    model_loaded = BooleanField()
    data = JSONField()


def home(request):
    return JsonResponse({
        'message': 'App is running',
        'status': 'ok',
    })


@extend_schema(
    request=PredictionInputSerializer,
    responses=PredictionResponseSerializer,
)
@api_view(['POST'])
def predict(request):
    incoming = request.data.copy() if hasattr(request.data, 'copy') else dict(request.data)
    field_aliases = {
        'education-num': 'education_num',
        'marital-status': 'marital_status',
        'capital-gain': 'capital_gain',
        'capital-loss': 'capital_loss',
        'hours-per-week': 'hours_per_week',
        'native-country': 'native_country',
    }
    for hyphen_key, underscore_key in field_aliases.items():
        if hyphen_key in incoming and underscore_key not in incoming:
            incoming[underscore_key] = incoming[hyphen_key]

    serializer = PredictionInputSerializer(data=incoming)
    serializer.is_valid(raise_exception=True)
    validated = serializer.validated_data

    payload = {
        'age': validated['age'],
        'workclass': validated['workclass'],
        'education-num': validated['education_num'],
        'marital-status': validated['marital_status'],
        'occupation': validated['occupation'],
        'relationship': validated['relationship'],
        'race': validated['race'],
        'sex': validated['sex'],
        'capital-gain': validated['capital_gain'],
        'capital-loss': validated['capital_loss'],
        'hours-per-week': validated['hours_per_week'],
        'native-country': validated['native_country'],
    }

    df = pd.DataFrame([payload])

    if MODEL is None:
        probability = 0.58 if payload['capital-gain'] + payload['capital-loss'] >= 5000 else 0.42
        prediction = _format_prediction('', probability)
    else:
        probability = float(MODEL.predict_proba(df)[0, 1])
        predicted_label = POSITIVE_LABEL if probability >= DECISION_THRESHOLD else '<=50K'
        prediction = _format_prediction(predicted_label, probability)

    response_data = {
        'prediction': prediction,
        'probability': round(probability, 3),
        'threshold': DECISION_THRESHOLD,
        'model_loaded': MODEL is not None,
        'data': payload,
    }
    return Response(response_data, status=status.HTTP_200_OK)
