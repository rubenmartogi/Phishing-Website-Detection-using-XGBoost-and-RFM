import 'dart:convert';
import 'package:http/http.dart' as http;

// Ganti IP sesuai server Flask kamu
const String BASE_URL = 'http://172.27.81.247:5000';

Future<Map<String, dynamic>> hybridPhishingDetection(String url) async {
  try {
    final response = await http.post(
      Uri.parse('$BASE_URL/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'url': url}),
    ).timeout(const Duration(seconds: 15));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;

      // Map response Flask → format yang dipakai phishing.dart
      final category = (data['category'] ?? '').toString().toLowerCase();
      final confidence = (data['confidence'] ?? data['final_prob'] ?? 0.0) as num;
      final label = data['label'] ?? data['result'] ?? '';
      final algorithm = data['algorithm'] ?? 'Hybrid (Rule + RF + XGB + LR)';

      String resultText;
      if (category == 'phishing' || label == 'phishing') {
        resultText = '🚨 PHISHING: URL ini berbahaya';
      } else if (category == 'suspicious') {
        resultText = '⚠️ SUSPICIOUS: URL mencurigakan';
      } else if (category == 'caution') {
        resultText = '🟡 CAUTION: Perlu perhatian';
      } else {
        resultText = '✅ SAFE: URL aman';
      }

      return {
        'result': resultText,
        'category': category.isEmpty ? label : category,
        'algorithm': algorithm,
        'confidence': confidence.toDouble(),
        'details': data['flags'] ?? data['details'] ?? [],
      };
    } else {
      final err = jsonDecode(response.body);
      return {
        'result': '❌ ERROR: ${err['error'] ?? response.statusCode}',
        'algorithm': 'Flask API',
        'confidence': 0.0,
        'category': 'error',
        'details': [],
      };
    }
  } catch (e) {
    return {
      'result': '❌ ERROR: Tidak dapat terhubung ke server',
      'algorithm': 'Flask API',
      'confidence': 0.0,
      'category': 'error',
      'details': [],
    };
  }
}