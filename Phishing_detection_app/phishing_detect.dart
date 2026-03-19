import 'dart:convert';
import 'package:http/http.dart' as http;

const String BASE_URL = 'http://172.27.81.247:5000';

Future<Map<String, dynamic>> hybridPhishingDetection(String url) async {
  try {
    final response = await http.post(
      Uri.parse('$BASE_URL/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'url': url,
      }),
    ).timeout(const Duration(seconds: 15));

    final data = jsonDecode(response.body) as Map<String, dynamic>;

    if (response.statusCode == 200) {
      final int finalLabel = (data['final_label'] ?? 0) as int;
      final double phishingProb = ((data['final_phishing_prob'] ?? 0.0) as num).toDouble();
      final double safeProb = ((data['final_safe_prob'] ?? (1.0 - phishingProb)) as num).toDouble();

      final bool isPhishing = finalLabel == 1;
      final String category = (data['category'] ?? (isPhishing ? 'phishing' : 'safe')).toString().toLowerCase();
      final double confidence = isPhishing ? phishingProb : safeProb;

      String resultText;
      if (isPhishing) {
        resultText = '🚨 PHISHING: Final probability ${(phishingProb * 100).toStringAsFixed(1)}%';
      } else {
        resultText = '✅ SAFE: Final probability ${(safeProb * 100).toStringAsFixed(1)}%';
      }

      return {
        'result': resultText,
        'category': category,
        'algorithm': 'Final Decision Probability (Rule + RF + XGB + Stacking)',
        'confidence': confidence,
        'details': [
          'Decision: ${data['decision_source'] ?? '-'}',
          'Risk category: ${data['risk_category'] ?? '-'}',
          'Mode: ${data['mode'] ?? '-'}',
        ],
      };
    } else {
      return {
        'result': '❌ ERROR: ${data['error'] ?? response.statusCode}',
        'algorithm': 'Flask API',
        'confidence': 0.0,
        'category': 'error',
        'details': [],
      };
    }
  } catch (_) {
    return {
      'result': '❌ ERROR: Tidak dapat terhubung ke server',
      'algorithm': 'Flask API',
      'confidence': 0.0,
      'category': 'error',
      'details': [],
    };
  }
}