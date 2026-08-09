import json
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'fraud_app' / 'model.pkl'
METADATA_PATH = BASE_DIR / 'artifacts' / 'model_metadata.json'

POSITIVE_LABEL = '>50K'
DEFAULT_THRESHOLD = 0.5

SAMPLE_PAYLOAD = {
    'age': 35,
    'workclass': 'Private',
    'education-num': 10,
    'marital-status': 'Never-married',
    'occupation': 'Other-service',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 1000,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States',
}


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


def home(request):
    return JsonResponse({
        'message': 'Fraud detection API is running',
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'model_path': str(MODEL_PATH),
        'decision_threshold': DECISION_THRESHOLD,
        'endpoints': {
            'health': {'method': 'GET', 'url': '/'},
            'predict': {
                'method': 'POST',
                'url': '/predict/',
                'content_type': 'application/json',
                'sample_body': SAMPLE_PAYLOAD,
            },
        },
    })


def _format_prediction(label, probability):
    is_fraud = label == POSITIVE_LABEL or probability >= DECISION_THRESHOLD
    return 'Likely fraud' if is_fraud else 'Likely safe'


def _parse_payload(request):
    content_type = (request.content_type or '').split(';')[0].strip().lower()
    if content_type == 'application/json':
        try:
            return json.loads(request.body.decode('utf-8')) if request.body else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON body: {exc}') from exc
    return request.POST.dict()


def _build_feature_row(payload):
    int_fields = ('age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week')
    data = {
        'age': payload.get('age', 0),
        'workclass': payload.get('workclass', 'Private'),
        'education-num': payload.get('education-num', 0),
        'marital-status': payload.get('marital-status', 'Never-married'),
        'occupation': payload.get('occupation', 'Other-service'),
        'relationship': payload.get('relationship', 'Not-in-family'),
        'race': payload.get('race', 'White'),
        'sex': payload.get('sex', 'Male'),
        'capital-gain': payload.get('capital-gain', 0),
        'capital-loss': payload.get('capital-loss', 0),
        'hours-per-week': payload.get('hours-per-week', 0),
        'native-country': payload.get('native-country', 'United-States'),
    }
    try:
        for field in int_fields:
            data[field] = int(data[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid numeric field: {exc}') from exc
    return data


@csrf_exempt
@require_http_methods(['POST'])
def predict(request):
    try:
        payload = _parse_payload(request)
        data = _build_feature_row(payload)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)

    df = pd.DataFrame([data])

    if MODEL is None:
        probability = 0.58 if data['capital-gain'] + data['capital-loss'] >= 5000 else 0.42
        prediction = _format_prediction('', probability)
    else:
        probability = float(MODEL.predict_proba(df)[0, 1])
        predicted_label = POSITIVE_LABEL if probability >= DECISION_THRESHOLD else '<=50K'
        prediction = _format_prediction(predicted_label, probability)

    return JsonResponse({
        'prediction': prediction,
        'probability': round(probability, 3),
        'threshold': DECISION_THRESHOLD,
        'model_loaded': MODEL is not None,
        'data': data,
    })
