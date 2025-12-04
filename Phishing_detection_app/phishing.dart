import 'package:ecell/phishing_detect.dart';
import 'package:flutter/material.dart';

class PhishingLink extends StatefulWidget {
  @override
  State<PhishingLink> createState() => _PhishingLinkState();
}

class _PhishingLinkState extends State<PhishingLink> with SingleTickerProviderStateMixin {
  late TextEditingController _urlController;
  late AnimationController _animController;
  String _result = '';
  bool _isLoading = false;
  Map<String, dynamic> _fullResult = {};
  double _confidence = 0.0;

  // Modern 2025 color palette
  static const Color primaryBlue = Color(0xFF0066FF);
  static const Color accentCyan = Color(0xFF00D9FF);
  static const Color darkBg = Color(0xFF0F1419);
  static const Color cardBg = Color(0xFF1A1F26);
  static const Color successGreen = Color(0xFF10B981);
  static const Color dangerRed = Color(0xFFEF4444);
  static const Color warningOrange = Color(0xFFF59E0B);

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController();
    _animController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 800),
    );
  }

  @override
  void dispose() {
    _urlController.dispose();
    _animController.dispose();
    super.dispose();
  }

  Future<void> _checkUrl() async {
    if (_urlController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please enter a URL'),
          backgroundColor: dangerRed,
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }

    setState(() => _isLoading = true);
    _animController.forward(from: 0.0);

    try {
      Map<String, dynamic> result = await hybridPhishingDetection(_urlController.text);
      setState(() {
        _result = result['result'] ?? 'No result';
        _fullResult = result;
        _confidence = (result['confidence'] ?? 0.0).toDouble();
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _result = 'Error: ${e.toString()}';
        _isLoading = false;
      });
    }
  }

  Color _getStatusColor() {
    String category = (_fullResult['category'] ?? '').toString().toLowerCase();
    if (category == 'phishing') return dangerRed;
    if (category == 'safe') return successGreen;
    if (category == 'suspicious') return warningOrange;
    return primaryBlue;
  }

  String _getStatusEmoji() {
    String category = (_fullResult['category'] ?? '').toString().toLowerCase();
    if (category == 'phishing') return '🚨';
    if (category == 'safe') return '✅';
    if (category == 'suspicious') return '⚠️';
    return '🔍';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: darkBg,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: darkBg,
        title: Row(
          children: [
            Container(
              padding: EdgeInsets.all(8),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [primaryBlue, accentCyan],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(Icons.security, color: Colors.white, size: 24),
            ),
            SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'SafeLink',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'URL Security Scanner',
                  style: TextStyle(color: accentCyan, fontSize: 11),
                ),
              ],
            )
          ],
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Input Section
              Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [cardBg, Color(0xFF252B35)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: primaryBlue.withOpacity(0.3), width: 1),
                  boxShadow: [
                    BoxShadow(
                      color: primaryBlue.withOpacity(0.1),
                      blurRadius: 20,
                      spreadRadius: 0,
                    ),
                  ],
                ),
                padding: EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Analyze URL',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 16),
                    TextField(
                      controller: _urlController,
                      enabled: !_isLoading,
                      style: TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: 'Enter URL (e.g., https://example.com)',
                        hintStyle: TextStyle(color: Colors.grey[600]),
                        prefixIcon: Icon(Icons.link, color: accentCyan),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: primaryBlue.withOpacity(0.5)),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: primaryBlue.withOpacity(0.3)),
                        ),
                        focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: accentCyan, width: 2),
                        ),
                        filled: true,
                        fillColor: Color(0xFF0F1419).withOpacity(0.6),
                      ),
                      onSubmitted: (_) => _checkUrl(),
                    ),
                    SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [primaryBlue, accentCyan],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(12),
                          boxShadow: [
                            BoxShadow(
                              color: primaryBlue.withOpacity(0.3),
                              blurRadius: 12,
                              spreadRadius: 0,
                            ),
                          ],
                        ),
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _checkUrl,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            disabledBackgroundColor: Colors.grey[700],
                          ),
                          child: _isLoading
                              ? SizedBox(
                                  width: 24,
                                  height: 24,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2.5,
                                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                  ),
                                )
                              : Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.search, size: 22),
                                    SizedBox(width: 8),
                                    Text(
                                      'Scan URL',
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 28),

              // Result Section
              if (_result.isNotEmpty)
                ScaleTransition(
                  scale: Tween<double>(begin: 0.8, end: 1.0).animate(
                    CurvedAnimation(parent: _animController, curve: Curves.elasticOut),
                  ),
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [cardBg, Color(0xFF252B35)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: _getStatusColor().withOpacity(0.4),
                        width: 2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: _getStatusColor().withOpacity(0.2),
                          blurRadius: 24,
                          spreadRadius: 0,
                        ),
                      ],
                    ),
                    padding: EdgeInsets.all(24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              _getStatusEmoji(),
                              style: TextStyle(fontSize: 32),
                            ),
                            SizedBox(width: 16),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Analysis Result',
                                    style: TextStyle(
                                      color: Colors.grey[500],
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 1,
                                    ),
                                  ),
                                  SizedBox(height: 4),
                                  Text(
                                    _result.replaceAll(RegExp(r'[🚨✅⚠️🔍]'), '').trim(),
                                    style: TextStyle(
                                      color: _getStatusColor(),
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 20),

                        // Confidence Bar
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'Confidence',
                                  style: TextStyle(
                                    color: Colors.grey[400],
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                Text(
                                  '${(_confidence * 100).toStringAsFixed(1)}%',
                                  style: TextStyle(
                                    color: accentCyan,
                                    fontSize: 13,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            SizedBox(height: 8),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: LinearProgressIndicator(
                                value: _confidence,
                                minHeight: 6,
                                backgroundColor: Colors.grey[800],
                                valueColor: AlwaysStoppedAnimation<Color>(_getStatusColor()),
                              ),
                            ),
                          ],
                        ),
                        SizedBox(height: 20),

                        // Details
                        if (_fullResult.isNotEmpty) ...[
                          Divider(color: Colors.grey[700], height: 24),
                          _buildDetailRow('URL', _urlController.text),
                          _buildDetailRow('Algorithm', _fullResult['algorithm'] ?? '-'),
                          _buildDetailRow('Category', _fullResult['category'] ?? '-'),
                          if (_fullResult['details'] != null)
                            _buildDetailsList('Flags', _fullResult['details']),
                        ],
                      ],
                    ),
                  ),
                ),

              // Info Cards
              SizedBox(height: 28),
              Text(
                'How It Works',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 12),
              _buildInfoCard(
                icon: Icons.rule,
                title: 'Rule-Based Detection',
                desc: 'Advanced pattern matching & domain analysis',
                color: primaryBlue,
              ),
              _buildInfoCard(
                icon: Icons.psychology,
                title: 'ML Models',
                desc: 'Random Forest + XGBoost ensemble',
                color: accentCyan,
              ),
              _buildInfoCard(
                icon: Icons.verified_user,
                title: 'Real-Time Analysis',
                desc: 'Instant URL safety verification',
                color: successGreen,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: TextStyle(
                color: Colors.grey[500],
                fontSize: 12,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.5,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailsList(String label, List<dynamic> items) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.grey[500],
              fontSize: 12,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
          SizedBox(height: 6),
          ...items.map((item) => Padding(
            padding: EdgeInsets.symmetric(vertical: 2),
            child: Row(
              children: [
                Icon(Icons.check_small, color: accentCyan, size: 16),
                SizedBox(width: 6),
                Expanded(
                  child: Text(
                    item.toString(),
                    style: TextStyle(color: Colors.grey[300], fontSize: 11),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildInfoCard({
    required IconData icon,
    required String title,
    required String desc,
    required Color color,
  }) {
    return Padding(
      padding: EdgeInsets.only(bottom: 12),
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [cardBg, Color(0xFF252B35)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.2), width: 1),
        ),
        padding: EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              padding: EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: color, size: 24),
            ),
            SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    desc,
                    style: TextStyle(
                      color: Colors.grey[500],
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}