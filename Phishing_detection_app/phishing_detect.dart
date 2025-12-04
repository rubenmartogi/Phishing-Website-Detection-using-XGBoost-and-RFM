import 'dart:math';
import 'dart:convert';
import 'package:http/http.dart' as http;

// Rule-based filtering constants
class PhishingRules {
  static const List<String> SUSPICIOUS_KEYWORDS = [
    'verify', 'update', 'suspend', 'confirm', 'secure', 'account',
    'login', 'bank', 'paypal', 'amazon', 'microsoft', 'apple',
    'urgent', 'expired', 'limited', 'click', 'winner'
  ];
  
  static const List<String> URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'short.link'
  ];
  
  static const List<String> LEGITIMATE_DOMAINS = [
    'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
    'apple.com', 'github.com', 'stackoverflow.com', 'wikipedia.org'
  ];
}

// Fungsi validasi domain agar hanya domain valid yang diproses
bool isValidDomain(String domain) {
  // Minimal: ada titik dan TLD 2-10 huruf
  final domainPattern = RegExp(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,10}$');
  return domainPattern.hasMatch(domain);
}

// Main hybrid detection function
Future<Map<String, dynamic>> hybridPhishingDetection(String url) async {
  try {
    Uri uri = Uri.parse(url);
    String domain = uri.host.toLowerCase();

    // Validasi domain
    if (!isValidDomain(domain)) {
      print('Invalid domain: $domain'); // Tambahkan ini untuk debug
      return {
        'result': '❌ ERROR: Invalid domain format',
        'algorithm': 'Validation',
        'confidence': 0.0,
      };
    }

    // Step 1: Rule-based Pre-filtering
    Map<String, dynamic> ruleResult = ruleBasedFiltering(url);
    
    if (ruleResult['confidence'] > 0.8) {
      // High confidence from rules, return immediately
      return ruleResult;
    }
    
    // Step 2: Extract features for ML models
    Map<String, double> features = await extractURLFeatures(url);
    
    // Step 3: Random Forest Classification
    Map<String, dynamic> rfResult = randomForestClassification(features);
    
    // Step 4: XGBoost Classification  
    Map<String, dynamic> xgbResult = xgboostClassification(features);
    
    // Step 5: Ensemble Decision
    return ensembleDecision(ruleResult, rfResult, xgbResult, url);
    
  } catch (e) {
    return {
      'result': '❌ ERROR: Unable to analyze URL',
      'algorithm': 'Error',
      'confidence': 0.0,
    };
  }
}

// Rule-based filtering (first filter)
Map<String, dynamic> ruleBasedFiltering(String url) {
  int riskScore = 0;
  List<String> detectedIssues = [];
  
  try {
    Uri uri = Uri.parse(url);
    String domain = uri.host.toLowerCase();
    String fullUrl = url.toLowerCase();
    
    // Rule 1: Check for legitimate domains (whitelist)
    for (String legit in PhishingRules.LEGITIMATE_DOMAINS) {
      if (domain.contains(legit)) {
        return {
          'result': '✅ SAFE: Legitimate domain detected',
          'algorithm': 'Rule-based (Whitelist)',
          'confidence': 0.95,
        };
      }
    }
    
    // Rule 2: IP Address instead of domain
    RegExp ipPattern = RegExp(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b');
    if (ipPattern.hasMatch(domain)) {
      riskScore += 3;
      detectedIssues.add('IP address in URL');
    }
    
    // Rule 3: @ symbol in URL
    if (url.contains('@')) {
      riskScore += 3;
      detectedIssues.add('@ symbol detected');
    }
    
    // Rule 4: URL length
    if (url.length > 100) {
      riskScore += 2;
      detectedIssues.add('Very long URL');
    } else if (url.length > 75) {
      riskScore += 1;
      detectedIssues.add('Long URL');
    }
    
    // Rule 5: Suspicious keywords
    int keywordCount = 0;
    for (String keyword in PhishingRules.SUSPICIOUS_KEYWORDS) {
      if (fullUrl.contains(keyword)) {
        keywordCount++;
      }
    }
    if (keywordCount >= 2) {
      riskScore += 2;
      detectedIssues.add('Multiple suspicious keywords');
    } else if (keywordCount >= 1) {
      riskScore += 1;
      detectedIssues.add('Suspicious keywords');
    }
    
    // Rule 6: URL shorteners
    for (String shortener in PhishingRules.URL_SHORTENERS) {
      if (domain.contains(shortener)) {
        riskScore += 2;
        detectedIssues.add('URL shortener detected');
        break;
      }
    }
    
    // Rule 7: Multiple subdomains
    List<String> parts = domain.split('.');
    if (parts.length > 4) {
      riskScore += 2;
      detectedIssues.add('Too many subdomains');
    } else if (parts.length > 3) {
      riskScore += 1;
      detectedIssues.add('Multiple subdomains');
    }
    
    // Rule 8: HTTPS check
    if (!url.startsWith('https://')) {
      riskScore += 1;
      detectedIssues.add('No HTTPS');
    }
    
    // Rule 9: Suspicious TLD
    List<String> suspiciousTLD = ['.tk', '.ml', '.ga', '.cf'];
    for (String tld in suspiciousTLD) {
      if (domain.endsWith(tld)) {
        riskScore += 2;
        detectedIssues.add('Suspicious TLD');
        break;
      }
    }
    
    // Decision based on risk score
    double confidence = min(riskScore / 10.0, 1.0);
    
    if (riskScore >= 7) {
      return {
        'result': '🚨 PHISHING: High risk detected (${detectedIssues.join(", ")})',
        'algorithm': 'Rule-based',
        'confidence': confidence,
      };
    } else if (riskScore >= 4) {
      return {
        'result': '⚠️ SUSPICIOUS: Multiple risk factors (${detectedIssues.join(", ")})',
        'algorithm': 'Rule-based',
        'confidence': confidence,
      };
    } else if (riskScore >= 1) {
      return {
        'result': '🟡 CAUTION: Some risk factors detected (${detectedIssues.join(", ")})',
        'algorithm': 'Rule-based',
        'confidence': confidence,
      };
    } else {
      return {
        'result': '✅ SAFE: No obvious threats detected',
        'algorithm': 'Rule-based',
        'confidence': max(0.3, 1.0 - confidence),
      };
    }
    
  } catch (e) {
    return {
      'result': '❌ ERROR: Invalid URL format',
      'algorithm': 'Rule-based',
      'confidence': 0.0,
    };
  }
}

// Extract 17 features for ML models
Future<Map<String, double>> extractURLFeatures(String url) async {
  Map<String, double> features = {};
  
  try {
    Uri uri = Uri.parse(url);
    String domain = uri.host;
    
    // Feature 1: Having IP Address
    RegExp ipPattern = RegExp(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b');
    features['having_IP'] = ipPattern.hasMatch(domain) ? 1.0 : 0.0;
    
    // Feature 2: URL Length
    features['URL_Length'] = url.length > 75 ? 1.0 : (url.length > 54 ? 0.0 : -1.0);
    
    // Feature 3: Having @ symbol
    features['having_At'] = url.contains('@') ? 1.0 : 0.0;
    
    // Feature 4: URL Depth
    int depth = uri.pathSegments.length;
    features['URL_Depth'] = depth > 3 ? 1.0 : (depth > 1 ? 0.0 : -1.0);
    
    // Feature 5: Redirection
    features['Redirection'] = url.split('//').length > 2 ? 1.0 : 0.0;
    
    // Feature 6: HTTPS Domain
    features['https_Domain'] = url.startsWith('https://') ? 0.0 : 1.0;
    
    // Feature 7: TinyURL
    List<String> shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co'];
    features['TinyURL'] = shorteners.any((s) => domain.contains(s)) ? 1.0 : 0.0;
    
    // Feature 8: Prefix/Suffix
    features['Prefix_Suffix'] = domain.contains('-') ? 1.0 : 0.0;
    
    // Simplified features for remaining 9 (would need actual web scraping)
    features['DNS_Record'] = 0.0; // Assume DNS exists
    features['Web_Traffic'] = 0.0; // Assume normal traffic
    features['Domain_Age'] = 0.0; // Assume old domain
    features['Domain_End'] = 0.0; // Assume long expiry
    features['iFrame'] = 0.0;
    features['Mouse_Over'] = 0.0;
    features['Right_Click'] = 0.0;
    features['Web_Forwards'] = 0.0;
    features['Statistical_Report'] = 0.0;
    
  } catch (e) {
    // Return default features on error
    for (int i = 0; i < 17; i++) {
      features['feature_$i'] = 0.0;
    }
  }
  
  return features;
}

// Simulated Random Forest Classification
Map<String, dynamic> randomForestClassification(Map<String, double> features) {
  // Simulate Random Forest decision based on key features
  double score = 0.0;
  int featureCount = 0;
  
  // Weight important features
  if (features['having_IP'] == 1.0) score += 0.15;
  if (features['having_At'] == 1.0) score += 0.15;
  if (features['URL_Length'] == 1.0) score += 0.10;
  if (features['https_Domain'] == 1.0) score += 0.10;
  if (features['TinyURL'] == 1.0) score += 0.12;
  if (features['Prefix_Suffix'] == 1.0) score += 0.08;
  if (features['URL_Depth'] == 1.0) score += 0.08;
  if (features['Redirection'] == 1.0) score += 0.12;
  
  // Add some randomness to simulate ensemble
  Random random = Random();
  score += (random.nextDouble() - 0.5) * 0.1;
  
  double confidence = min(max(score, 0.0), 1.0);
  
  if (score > 0.6) {
    return {
      'prediction': 'phishing',
      'confidence': confidence,
      'algorithm': 'Random Forest'
    };
  } else {
    return {
      'prediction': 'legitimate',
      'confidence': 1.0 - confidence,
      'algorithm': 'Random Forest'
    };
  }
}

// Simulated XGBoost Classification
Map<String, dynamic> xgboostClassification(Map<String, double> features) {
  // Simulate XGBoost with different weights
  double score = 0.0;
  
  // XGBoost typically gives different weights
  if (features['having_IP'] == 1.0) score += 0.18;
  if (features['having_At'] == 1.0) score += 0.16;
  if (features['URL_Length'] == 1.0) score += 0.12;
  if (features['https_Domain'] == 1.0) score += 0.08;
  if (features['TinyURL'] == 1.0) score += 0.14;
  if (features['Prefix_Suffix'] == 1.0) score += 0.06;
  if (features['URL_Depth'] == 1.0) score += 0.10;
  if (features['Redirection'] == 1.0) score += 0.14;
  
  // XGBoost gradient boosting simulation
  Random random = Random();
  score += (random.nextDouble() - 0.5) * 0.05;
  
  double confidence = min(max(score, 0.0), 1.0);
  
  if (score > 0.55) {
    return {
      'prediction': 'phishing',
      'confidence': confidence,
      'algorithm': 'XGBoost'
    };
  } else {
    return {
      'prediction': 'legitimate',
      'confidence': 1.0 - confidence,
      'algorithm': 'XGBoost'
    };
  }
}

// Ensemble decision combining all three approaches
Map<String, dynamic> ensembleDecision(
  Map<String, dynamic> ruleResult,
  Map<String, dynamic> rfResult, 
  Map<String, dynamic> xgbResult,
  String url
) {
  // Weighted voting system
  double phishingVotes = 0.0;
  double legitimateVotes = 0.0;
  
  List<String> algorithmsUsed = [];
  
  // Rule-based weight: 30%
  if (ruleResult['result'].contains('PHISHING') || ruleResult['result'].contains('🚨')) {
    phishingVotes += 0.3 * ruleResult['confidence'];
  } else {
    legitimateVotes += 0.3 * ruleResult['confidence'];
  }
  algorithmsUsed.add('Rules');
  
  // Random Forest weight: 35%
  if (rfResult['prediction'] == 'phishing') {
    phishingVotes += 0.35 * rfResult['confidence'];
  } else {
    legitimateVotes += 0.35 * rfResult['confidence'];
  }
  algorithmsUsed.add('RF');
  
  // XGBoost weight: 35%
  if (xgbResult['prediction'] == 'phishing') {
    phishingVotes += 0.35 * xgbResult['confidence'];
  } else {
    legitimateVotes += 0.35 * xgbResult['confidence'];
  }
  algorithmsUsed.add('XGB');
  
  // Final decision
  double totalVotes = phishingVotes + legitimateVotes;
  double confidence = max(phishingVotes, legitimateVotes) / totalVotes;
  
  String result;
  if (phishingVotes > legitimateVotes) {
    if (confidence > 0.8) {
      result = '🚨 PHISHING: High confidence detection';
    } else if (confidence > 0.6) {
      result = '⚠️ LIKELY PHISHING: Moderate confidence';
    } else {
      result = '🟡 SUSPICIOUS: Low confidence phishing';
    }
  } else {
    if (confidence > 0.8) {
      result = '✅ SAFE: High confidence legitimate';
    } else if (confidence > 0.6) {
      result = '✅ LIKELY SAFE: Moderate confidence';
    } else {
      result = '🟡 UNCERTAIN: Low confidence legitimate';
    }
  }
  
  return {
    'result': result,
    'algorithm': 'Hybrid (${algorithmsUsed.join(" + ")})',
    'confidence': confidence,
  };
}

// Legacy function for backward compatibility
Future<String> check_website_status(String url) async {
  Map<String, dynamic> result = await hybridPhishingDetection(url);
  return result['result'];
}