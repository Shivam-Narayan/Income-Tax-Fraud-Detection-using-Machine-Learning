import json

from django.test import SimpleTestCase
from django.urls import reverse


class FraudAppTests(SimpleTestCase):
    def test_home_endpoint(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_prediction_form_post(self):
        response = self.client.post(reverse('predict'), {
            'age': '35',
            'education-num': '10',
            'capital-gain': '1000',
            'capital-loss': '0',
            'hours-per-week': '40',
            'sex': 'Male',
            'workclass': 'Private',
            'marital-status': 'Never-married',
            'occupation': 'Other-service',
            'relationship': 'Not-in-family',
            'race': 'White',
            'native-country': 'United-States',
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body['prediction'], ('Likely fraud', 'Likely safe'))
        self.assertIsNotNone(body['probability'])

    def test_prediction_json_post(self):
        response = self.client.post(
            reverse('predict'),
            data=json.dumps({
                'age': 44,
                'workclass': 'Private',
                'education-num': 13,
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 7688,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States',
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['model_loaded'])
        self.assertIn('probability', body)

    def test_prediction_rejects_missing_required_fields(self):
        response = self.client.post(reverse('predict'), {'age': '35'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('workclass', response.json())
