#!/usr/bin/env python3
"""
GDPR Breach Risk Assessment Tool
Usage: python3 breach_risk_cli.py
"""

def get_breach_input():
    """Interactive CLI to gather breach characteristics"""
    print("=== GDPR Breach Risk Assessment ===\n")

    characteristics = {}

    # Country
    country = input("Country code (ES/IT/RO/GB/FR/PL/etc.): ").strip().upper()
    characteristics["country"] = country

    # Initiation channel
    print("\nInitiation channel:")
    print("1. COMPLAINT")
    print("2. BREACH_NOTIFICATION")
    print("3. EX_OFFICIO_DPA_INITIATIVE")
    channel_choice = input("Choose (1-3): ").strip()

    channel_map = {
        "1": "COMPLAINT",
        "2": "BREACH_NOTIFICATION",
        "3": "EX_OFFICIO_DPA_INITIATIVE"
    }
    characteristics["initiation_channel"] = channel_map.get(channel_choice, "UNKNOWN")

    # Timing compliance
    timing = input("\nNotified DPA within 72 hours? (y/n): ").strip().lower()
    characteristics["timing_compliant"] = timing.startswith('y')

    # Subject notification
    subjects = input("Data subjects notified? (y/n): ").strip().lower()
    characteristics["subjects_notified"] = subjects.startswith('y')

    notification_req = input("Subject notification required? (y/n): ").strip().lower()
    characteristics["notification_required"] = notification_req.startswith('y')

    # Vulnerable subjects
    vulnerable = input("Vulnerable subjects involved? (y/n): ").strip().lower()
    characteristics["vulnerable_subjects"] = vulnerable.startswith('y')

    # Remedial actions
    remedial = input("Proactive remedial actions taken? (y/n): ").strip().lower()
    characteristics["remedial_actions"] = remedial.startswith('y')

    return characteristics

def main():
    """Main CLI interface"""
    characteristics = get_breach_input()

    print("\n" + "="*50)
    print("RISK ASSESSMENT RESULTS")
    print("="*50)

    # This would integrate with the full analysis
    # For demo purposes, showing structure:

    print(f"Country: {characteristics['country']}")
    print(f"Initiation: {characteristics['initiation_channel']}")
    print(f"Timing compliant: {characteristics['timing_compliant']}")
    print(f"Subjects notified: {characteristics['subjects_notified']}")

    # Placeholder recommendation logic
    risk_score = 0

    if not characteristics["timing_compliant"]:
        risk_score += 30
    if characteristics["vulnerable_subjects"]:
        risk_score += 25
    if characteristics["country"] in ["RO", "PL"]:
        risk_score += 35
    if characteristics["initiation_channel"] == "EX_OFFICIO_DPA_INITIATIVE":
        risk_score += 20

    if characteristics["remedial_actions"]:
        risk_score -= 15
    if characteristics["subjects_notified"]:
        risk_score -= 10

    if risk_score >= 70:
        recommendation = "FILE_NOW"
        confidence = "HIGH"
    elif risk_score >= 40:
        recommendation = "INITIAL_NOTICE"
        confidence = "MEDIUM"
    else:
        recommendation = "DOCUMENT_ONLY"
        confidence = "LOW"

    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"CONFIDENCE: {confidence}")
    print(f"RISK SCORE: {risk_score}/100")

    print("\n⚠️  DISCLAIMER: This tool provides guidance based on statistical")
    print("analysis of past decisions. Consult legal counsel for specific cases.")

if __name__ == "__main__":
    main()
