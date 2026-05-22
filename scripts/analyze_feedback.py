"""
AI Analyst End-to-End Feedback Analysis Pipeline
=================================================================================
Runs complete workflow:
1. Data profiling & quality checks
2. Segmentation analysis (Positive/Negative, by Region)
3. Comment field text mining (semantic clustering - Option B)
4. Impact scoring 
5. Narrative synthesis
6. Report generation

Usage:
    python scripts/analyze_feedback.py
    
Output files:
    - outputs/feedback_analysis_summary.md
    - outputs/feedback_findings_ranked.csv
    - outputs/top_issues_text_mining.csv
    - outputs/segment_comparison.csv
    - outputs/feedback_quality_report.csv
"""

import sys
from pathlib import Path
from datetime import datetime
import re
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from scipy import spatial
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.analytics_helpers import (
    score_findings,
    synthesize_insights,
    concentration_analysis,
)
from helpers.data_helpers import read_table, list_tables


# =============================================================================
# STEP 1: DATA LOADING & PROFILING
# =============================================================================

def load_feedback_data(filepath):
    """Load feedback CSV and return DataFrame."""
    print("\n📂 STEP 1: LOADING DATA")
    print(f"   Loading from: {filepath}")
    
    df = pd.read_csv(filepath, low_memory=False)
    print(f"   ✓ Loaded {len(df):,} records × {len(df.columns)} columns")
    
    # Convert timestamp
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    
    return df


def profile_data(df):
    """Data quality checks and profiling."""
    print("\n📊 STEP 2: DATA QUALITY CHECK")
    
    profile = {
        "total_records": len(df),
        "date_range": None,
        "critical_fields_populated": {},
        "nulls_by_column": {},
        "unique_values": {},
    }
    
    # Date range
    if "event_timestamp" in df.columns:
        valid_dates = df["event_timestamp"].dropna()
        if len(valid_dates) > 0:
            profile["date_range"] = {
                "min": valid_dates.min(),
                "max": valid_dates.max(),
            }
    
    # Critical fields
    critical_fields = ["feedback", "reasons", "comment", "account_name", "region"]
    for field in critical_fields:
        if field in df.columns:
            nulls = df[field].isna().sum()
            populated = len(df) - nulls
            pct = (populated / len(df)) * 100 if len(df) > 0 else 0
            profile["critical_fields_populated"][field] = {
                "populated": populated,
                "nulls": nulls,
                "pct_populated": round(pct, 1),
            }
            print(f"   {field:20s}: {pct:5.1f}% populated ({populated:,} records)")
    
    # Feedback distribution
    if "feedback" in df.columns:
        feedback_dist = df["feedback"].value_counts().to_dict()
        print(f"\n   Feedback distribution:")
        for sentiment, count in sorted(feedback_dist.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df)) * 100
            print(f"      {sentiment:15s}: {count:5d} ({pct:5.1f}%)")
    
    # Region distribution
    if "region" in df.columns:
        region_dist = df["region"].value_counts()
        print(f"\n   Top regions:")
        for region, count in region_dist.head(5).items():
            pct = (count / len(df)) * 100
            print(f"      {region:20s}: {count:5d} ({pct:5.1f}%)")
    
    return profile


# =============================================================================
# STEP 3: SEGMENTATION ANALYSIS
# =============================================================================

def segment_by_sentiment(df):
    """Compare Positive vs Negative feedback."""
    print("\n📈 STEP 3A: SEGMENTATION - FEEDBACK SENTIMENT")
    
    segments = {}
    for sentiment in ["Positive", "Negative"]:
        segment_df = df[df["feedback"] == sentiment]
        
        if len(segment_df) == 0:
            continue
        
        comment_lengths = segment_df["comment"].fillna("").apply(len)
        
        result = {
            "count": len(segment_df),
            "pct": round((len(segment_df) / len(df)) * 100, 1),
            "avg_comment_length": round(comment_lengths.mean(), 1),
            "median_comment_length": int(comment_lengths.median()),
            "num_accounts": segment_df["account_name"].nunique(),
            "num_regions": segment_df["region"].nunique(),
            "num_cisco_users": (segment_df["cisco_user"] == "Yes").sum(),
        }
        
        segments[sentiment] = result
        
        print(f"\n   {sentiment} Feedback:")
        print(f"      Count: {result['count']:,} ({result['pct']:.1f}%)")
        print(f"      Avg comment length: {result['avg_comment_length']:.0f} chars")
        print(f"      Unique accounts: {result['num_accounts']:,}")
        print(f"      Regions: {result['num_regions']:,}")
        print(f"      Cisco users: {result['num_cisco_users']:,}")
    
    return segments


def segment_by_region(df):
    """Compare metrics by region."""
    print("\n📈 STEP 3B: SEGMENTATION - BY REGION")
    
    region_analysis = []
    
    for region in df["region"].unique():
        if pd.isna(region):
            continue
        
        region_df = df[df["region"] == region]
        
        positive_count = (region_df["feedback"] == "Positive").sum()
        negative_count = (region_df["feedback"] == "Negative").sum()
        
        result = {
            "region": region,
            "total_feedback": len(region_df),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_pct": round((positive_count / len(region_df)) * 100, 1) if len(region_df) > 0 else 0,
            "unique_accounts": region_df["account_name"].nunique(),
        }
        region_analysis.append(result)
    
    region_df = pd.DataFrame(region_analysis).sort_values("total_feedback", ascending=False)
    
    print(f"\n   Top regions by feedback volume:")
    for _, row in region_df.head(5).iterrows():
        print(f"      {row['region']:20s}: {row['total_feedback']:4d} total "
              f"({row['positive_pct']:5.1f}% positive from {row['unique_accounts']:,} accounts)")
    
    return region_df


# =============================================================================
# STEP 4: COMMENT FIELD TEXT MINING (OPTION B - SEMANTIC CLUSTERING)
# =============================================================================

def extract_noun_phrases(text):
    """Simple noun phrase extraction using regex patterns."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    text = text.lower().strip()
    
    # Remove common stop words and short words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "to",
        "of", "in", "on", "at", "by", "for", "with", "as", "it", "that",
        "this", "which", "who", "when", "where", "why", "how", "from",
        "up", "out", "if", "about", "no", "not", "so", "some", "very",
    }
    
    # Extract phrases: sequences of words separated by spaces/hyphens
    phrases = re.findall(r'\b[a-z\-]+(?:\s+[a-z\-]+){0,2}\b', text)
    phrases = [p.strip() for p in phrases if len(p) > 2]
    
    # Filter out stop words and very common terms
    phrases = [p for p in phrases if p not in stop_words and len(p.split()) >= 1]
    
    return phrases[:3]  # Return top 3 phrases per comment


def semantic_cluster_comments(comments_series, n_clusters=10):
    """Cluster comments using TF-IDF + hierarchical clustering.
    
    This is Option B (semantic-based) analysis.
    """
    print("\n💭 STEP 4: COMMENT FIELD ANALYSIS (Text Mining - Semantic Clustering)")
    
    # Filter out empty comments
    comments = [str(c).strip() for c in comments_series if pd.notna(c) and str(c).strip()]
    
    if len(comments) == 0:
        print("   ⚠ No comments found for analysis")
        return []
    
    print(f"   Analyzing {len(comments):,} comments...")
    
    # TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=2,  # Appear in at least 2 comments
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words='english',
        )
        tfidf_matrix = vectorizer.fit_transform(comments)
        
        if tfidf_matrix.shape[0] < 2:
            print("   ⚠ Not enough comments for clustering")
            return []
        
        # Compute distance matrix
        from sklearn.metrics.pairwise import cosine_distances
        distance_matrix = cosine_distances(tfidf_matrix)
        
        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)], method='ward')
        
        # Determine optimal number of clusters
        n_clusters = min(n_clusters, len(comments) // 2)
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group comments by cluster
        clustered = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            clustered[cluster_id].append(comments[idx])
        
        # Extract themes from each cluster
        themes = []
        for cluster_id in sorted(clustered.keys()):
            cluster_comments = clustered[cluster_id]
            
            # Extract representative phrases
            all_phrases = []
            for comment in cluster_comments:
                all_phrases.extend(extract_noun_phrases(comment))
            
            phrase_counts = Counter(all_phrases)
            top_phrases = [p for p, _ in phrase_counts.most_common(3)]
            
            # Pick most representative comment (longest one with most top phrases)
            representative = max(
                cluster_comments,
                key=lambda c: sum(1 for p in top_phrases if p in c.lower())
            )
            
            themes.append({
                "cluster_id": cluster_id,
                "theme": " | ".join(top_phrases) if top_phrases else "General feedback",
                "num_comments": len(cluster_comments),
                "representative_quote": representative[:150] + "..." if len(representative) > 150 else representative,
            })
        
        # Sort by frequency
        themes = sorted(themes, key=lambda x: x["num_comments"], reverse=True)
        
        print(f"\n   Identified {len(themes)} semantic clusters:")
        for theme in themes[:10]:
            pct = (theme["num_comments"] / len(comments)) * 100
            print(f"      Cluster {theme['cluster_id']:2d}: {theme['theme']:50s} "
                  f"({theme['num_comments']:3d} comments, {pct:4.1f}%)")
        
        return themes
    
    except Exception as e:
        print(f"   ⚠ Clustering failed: {e}")
        return []


def extract_issues_from_comments(df, negative_only=True):
    """Extract actionable issues from comments."""
    print("\n🔎 STEP 4B: EXTRACTING ISSUES FROM COMMENTS")
    
    if negative_only:
        comments_df = df[df["feedback"] == "Negative"].copy()
        print(f"   Analyzing {len(comments_df):,} negative feedback comments")
    else:
        comments_df = df.copy()
        print(f"   Analyzing {len(df):,} all feedback comments")
    
    # Semantic clustering
    semantic_themes = semantic_cluster_comments(comments_df["comment"], n_clusters=12)
    
    # Issue categorization patterns
    issue_patterns = {
        "Performance": r"(slow|lag|delay|timeout|hang|freeze|wait|performance|speed|fast)",
        "Bugs": r"(bug|crash|error|fail|broken|not work|doesn't work|doesn't load|failed to)",
        "UI/UX": r"(button|click|confusing|unclear|hard to find|where is|how do i|difficult|complicated)",
        "Feature Request": r"(need|want|should have|would like|feature|add|request|wish|could|would be nice)",
        "Integration": r"(integration|api|connect|sync|export|import|webhook|third-party)",
        "Documentation": r"(documentation|help|tutorial|guide|instruction|how to|manual)",
        "Data/Reporting": r"(data|report|analytics|metric|number|dashboard|accuracy|missing data)",
    }
    
    # Categorize semantic themes
    categorized_issues = []
    for theme in semantic_themes:
        theme_text = theme["theme"].lower()
        
        # Determine primary category
        category = "Other"
        for cat, pattern in issue_patterns.items():
            if re.search(pattern, theme_text):
                category = cat
                break
        
        issue = {
            "issue_id": theme["cluster_id"],
            "issue_description": theme["theme"],
            "issue_category": category,
            "frequency": theme["num_comments"],
            "representative_quote": theme["representative_quote"],
            "affected_accounts": 0,  # Will fill below
            "severity_score": 0,  # Will fill below
        }
        categorized_issues.append(issue)
    
    # Link issues to accounts
    for issue in categorized_issues:
        issue_text = issue["issue_description"].lower()
        matching_rows = comments_df[
            comments_df["comment"].fillna("").apply(
                lambda c: any(phrase in c.lower() for phrase in issue_text.split("|"))
            )
        ]
        
        if len(matching_rows) > 0:
            issue["affected_accounts"] = matching_rows["account_name"].nunique()
        
        # Basic severity: frequency + account spread
        frequency_score = min(issue["frequency"] / len(comments_df) * 100 * 2, 50)
        account_score = min(issue["affected_accounts"] / df["account_name"].nunique() * 100, 50)
        issue["severity_score"] = round(frequency_score + account_score, 1)
    
    # Sort by severity
    categorized_issues = sorted(categorized_issues, key=lambda x: x["severity_score"], reverse=True)
    
    print(f"\n   Extracted {len(categorized_issues)} issues:")
    for issue in categorized_issues[:8]:
        print(f"      [{issue['issue_category']:12s}] {issue['issue_description']:45s} "
              f"({issue['frequency']:3d} comments, {issue['affected_accounts']:2d} accounts)")
    
    return categorized_issues


# =============================================================================
# STEP 5: IMPACT SCORING & PRIORITIZATION
# =============================================================================

def score_feedback_findings(issues_list, accounts_df, total_records):
    """Score findings using 4-factor model: Magnitude, Breadth, Actionability, Confidence."""
    print("\n⭐ STEP 5: IMPACT SCORING (4-Factor Model)")
    
    findings = []
    
    for issue in issues_list:
        # Magnitude: relative frequency of issue
        magnitude_score = min(
            (issue["frequency"] / total_records * 100) * 5,  # Scale 0-100
            100
        )
        
        # Breadth: number of affected accounts
        breadth_score = min(
            (issue["affected_accounts"] / accounts_df["account_name"].nunique() * 100) * 2,
            100
        )
        
        # Actionability: how specific is the issue? (semantic description length)
        description_length = len(issue["issue_description"].split())
        actionability_score = min(description_length * 10, 100)
        
        # Confidence: based on frequency and account spread
        frequency_confidence = min(issue["frequency"] / 5, 100)
        account_confidence = min(issue["affected_accounts"] / 2, 100)
        confidence_score = (frequency_confidence + account_confidence) / 2
        
        # Overall impact score (composite)
        impact_score = round(
            (magnitude_score * 0.4 + breadth_score * 0.3 + actionability_score * 0.2 + confidence_score * 0.1),
            1
        )
        
        finding = {
            "description": issue["issue_description"],
            "category": issue["issue_category"],
            "frequency": issue["frequency"],
            "affected_accounts": issue["affected_accounts"],
            "representative_quote": issue["representative_quote"],
            "magnitude_score": round(magnitude_score, 1),
            "breadth_score": round(breadth_score, 1),
            "actionability_score": round(actionability_score, 1),
            "confidence_score": round(confidence_score, 1),
            "impact_score": impact_score,
        }
        findings.append(finding)
    
    # Sort by impact
    findings = sorted(findings, key=lambda x: x["impact_score"], reverse=True)
    
    print(f"\n   Top 10 findings by impact score:")
    for i, finding in enumerate(findings[:10], 1):
        print(f"      {i:2d}. [{finding['category']:12s}] {finding['description']:45s} "
              f"Impact: {finding['impact_score']:6.1f}")
    
    return findings


# =============================================================================
# STEP 6: NARRATIVE SYNTHESIS
# =============================================================================

def synthesize_feedback_narrative(df, segments, issues, regional_analysis):
    """Generate narrative summary of findings."""
    print("\n📝 STEP 6: NARRATIVE SYNTHESIS")
    
    narrative = {
        "title": "Cisco IQ Feedback Analysis",
        "executive_summary": "",
        "key_metrics": {},
        "findings_by_category": {},
        "recommendations": [],
    }
    
    # Key metrics
    total_feedback = len(df)
    date_range = df["event_timestamp"].min() if "event_timestamp" in df.columns else None
    
    positive_count = (df["feedback"] == "Positive").sum()
    positive_pct = (positive_count / total_feedback * 100) if total_feedback > 0 else 0
    
    narrative["key_metrics"] = {
        "total_feedback_records": total_feedback,
        "date_range": f"{date_range.date()}" if date_range else "Unknown",
        "positive_feedback_count": positive_count,
        "positive_feedback_pct": round(positive_pct, 1),
        "unique_accounts": df["account_name"].nunique(),
        "regions_covered": df["region"].nunique(),
    }
    
    # Executive summary
    narrative["executive_summary"] = (
        f"Analysis of {total_feedback:,} Cisco IQ user feedback submissions revealed "
        f"{positive_pct:.1f}% positive sentiment, with key pain points concentrated in "
        f"{len(issues) if issues else 0} distinct issue areas across "
        f"{narrative['key_metrics']['unique_accounts']:,} accounts spanning "
        f"{narrative['key_metrics']['regions_covered']} regions. "
        f"Top issues involve {issues[0]['category'] if issues else 'general'} concerns, "
        f"affecting {issues[0]['affected_accounts'] if issues else 0} accounts."
    )
    
    # Group findings by category
    findings_by_cat = defaultdict(list)
    for issue in issues[:15]:  # Top 15
        findings_by_cat[issue["category"]].append(issue)
    
    for category, category_issues in sorted(findings_by_cat.items()):
        narrative["findings_by_category"][category] = [
            {
                "description": i["description"],
                "frequency": i["frequency"],
                "accounts": i["affected_accounts"],
                "impact": i["impact_score"],
            }
            for i in category_issues[:3]
        ]
    
    # Generate recommendations
    for i, issue in enumerate(issues[:5], 1):
        recommendation = {
            "priority": i,
            "issue": issue["description"],
            "category": issue["category"],
            "action": generate_recommendation(issue),
            "impact_score": issue["impact_score"],
        }
        narrative["recommendations"].append(recommendation)
    
    print(f"\n   ✓ Generated narrative with {len(narrative['recommendations'])} recommendations")
    print(f"   ✓ {len(narrative['findings_by_category'])} issue categories identified")
    
    return narrative


def generate_recommendation(issue):
    """Generate actionable recommendation based on issue."""
    if issue["category"] == "Performance":
        return f"Investigate and optimize {issue['description']} to improve user experience"
    elif issue["category"] == "Bugs":
        return f"Prioritize bug fix for {issue['description']} affecting {issue['affected_accounts']} accounts"
    elif issue["category"] == "UI/UX":
        return f"Conduct UX review of {issue['description']} based on {issue['frequency']} user complaints"
    elif issue["category"] == "Feature Request":
        return f"Evaluate feature request: {issue['description']} (requested in {issue['frequency']} submissions)"
    else:
        return f"Address {issue['category'].lower()} issue: {issue['description']}"


# =============================================================================
# STEP 7: REPORT GENERATION
# =============================================================================

def export_findings_csv(findings, filepath):
    """Export scored findings to CSV."""
    df = pd.DataFrame(findings)
    df = df[[
        "description", "category", "frequency", "affected_accounts",
        "magnitude_score", "breadth_score", "actionability_score",
        "confidence_score", "impact_score", "representative_quote"
    ]].copy()
    
    df = df.rename(columns={
        "description": "Issue Description",
        "category": "Category",
        "frequency": "Frequency",
        "affected_accounts": "Affected Accounts",
        "magnitude_score": "Magnitude Score",
        "breadth_score": "Breadth Score",
        "actionability_score": "Actionability Score",
        "confidence_score": "Confidence Score",
        "impact_score": "Impact Score",
        "representative_quote": "Representative Quote",
    })
    
    df.to_csv(filepath, index=False)
    print(f"   ✓ Exported findings to: {filepath}")


def export_segment_analysis_csv(segments_by_sentiment, region_analysis, filepath):
    """Export segment analysis to CSV."""
    output_rows = []
    
    for sentiment, metrics in segments_by_sentiment.items():
        output_rows.append({
            "Segment Type": "Sentiment",
            "Segment Value": sentiment,
            "Count": metrics["count"],
            "Percentage": f"{metrics['pct']}%",
            "Avg Comment Length": metrics["avg_comment_length"],
            "Unique Accounts": metrics["num_accounts"],
            "Regions": metrics["num_regions"],
            "Cisco Users": metrics["num_cisco_users"],
        })
    
    for _, row in region_analysis.iterrows():
        output_rows.append({
            "Segment Type": "Region",
            "Segment Value": row["region"],
            "Count": row["total_feedback"],
            "Percentage": f"{row['positive_pct']}% positive",
            "Avg Comment Length": "N/A",
            "Unique Accounts": row["unique_accounts"],
            "Regions": 1,
            "Cisco Users": "N/A",
        })
    
    df = pd.DataFrame(output_rows)
    df.to_csv(filepath, index=False)
    print(f"   ✓ Exported segment analysis to: {filepath}")


def export_quality_report(profile, filepath):
    """Export data quality report."""
    rows = []
    
    for field, stats in profile["critical_fields_populated"].items():
        rows.append({
            "Field": field,
            "Populated": stats["populated"],
            "Nulls": stats["nulls"],
            "% Populated": f"{stats['pct_populated']:.1f}%",
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"   ✓ Exported quality report to: {filepath}")


def export_narrative_markdown(narrative, filepath):
    """Export narrative summary as markdown."""
    content = f"""# Cisco IQ Feedback Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{narrative['executive_summary']}

## Key Metrics
- **Total Feedback Records**: {narrative['key_metrics']['total_feedback_records']:,}
- **Date Range**: {narrative['key_metrics']['date_range']}
- **Positive Feedback**: {narrative['key_metrics']['positive_feedback_count']:,} ({narrative['key_metrics']['positive_feedback_pct']:.1f}%)
- **Unique Accounts**: {narrative['key_metrics']['unique_accounts']:,}
- **Regions Covered**: {narrative['key_metrics']['regions_covered']}

## Top Issues by Category
"""
    
    for category, issues in narrative["findings_by_category"].items():
        content += f"\n### {category}\n"
        for issue in issues:
            content += f"- **{issue['description']}** ({issue['frequency']} mentions, {issue['accounts']} accounts, Impact: {issue['impact']:.1f})\n"
    
    content += "\n## Recommended Actions (Priority Order)\n"
    for rec in narrative["recommendations"]:
        content += f"{rec['priority']}. **[{rec['category']}]** {rec['action']}\n"
        content += f"   - Issue: {rec['issue']}\n"
        content += f"   - Impact Score: {rec['impact_score']:.1f}\n\n"
    
    with open(filepath, "w") as f:
        f.write(content)
    
    print(f"   ✓ Exported narrative to: {filepath}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete feedback analysis pipeline."""
    print("\n" + "="*80)
    print("AI ANALYST FEEDBACK ANALYSIS - END-TO-END PIPELINE")
    print("="*80)
    
    # Configure paths
    data_dir = Path(__file__).parent.parent / "data" / "my_experiment"
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Data file
    csv_file = data_dir / "webex_iq_feedback_messages.csv"
    
    if not csv_file.exists():
        print(f"❌ ERROR: {csv_file} not found")
        print(f"   Expected location: {csv_file}")
        sys.exit(1)
    
    # Step 1: Load data
    df = load_feedback_data(str(csv_file))
    
    # Step 2: Profile
    profile = profile_data(df)
    
    # Step 3: Segmentation
    segments_by_sentiment = segment_by_sentiment(df)
    region_analysis = segment_by_region(df)
    
    # Step 4: Comment analysis
    issues = extract_issues_from_comments(df, negative_only=True)
    
    # Step 5: Scoring
    scored_findings = score_feedback_findings(issues, df[["account_name"]].drop_duplicates(), len(df))
    
    # Step 6: Narrative
    narrative = synthesize_feedback_narrative(df, segments_by_sentiment, scored_findings, region_analysis)
    
    # Step 7: Export
    print("\n💾 STEP 7: EXPORTING RESULTS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_findings_csv(
        scored_findings,
        output_dir / f"feedback_findings_ranked_{timestamp}.csv"
    )
    
    export_segment_analysis_csv(
        segments_by_sentiment,
        region_analysis,
        output_dir / f"segment_comparison_{timestamp}.csv"
    )
    
    export_quality_report(
        profile,
        output_dir / f"feedback_quality_report_{timestamp}.csv"
    )
    
    export_narrative_markdown(
        narrative,
        output_dir / f"feedback_analysis_summary_{timestamp}.md"
    )
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total feedback analyzed: {len(df):,}")
    print(f"   Issues identified: {len(scored_findings)}")
    print(f"   Recommendations generated: {len(narrative['recommendations'])}")
    print(f"\n📁 Output files created in: {output_dir}")
    print(f"   - feedback_findings_ranked_{timestamp}.csv")
    print(f"   - segment_comparison_{timestamp}.csv")
    print(f"   - feedback_quality_report_{timestamp}.csv")
    print(f"   - feedback_analysis_summary_{timestamp}.md")


if __name__ == "__main__":
    main()