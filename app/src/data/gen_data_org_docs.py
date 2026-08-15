import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Create target directory
OUTPUT_DIR = "documents"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_styles():
    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1B365D")  # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Slate Blue

    styles.add(
        ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=primary_color,
            spaceAfter=6,
            bold=True,
        )
    )

    styles.add(
        ParagraphStyle(
            "SubTitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#718096"),
            spaceAfter=15,
        )
    )

    styles.add(
        ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=secondary_color,
            spaceBefore=12,
            spaceAfter=6,
            bold=True,
        )
    )

    styles.add(
        ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2D3748"),
            spaceAfter=8,
        )
    )

    return styles


def create_pdf(filename, elements):
    filepath = os.path.join(OUTPUT_DIR, filename)
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    doc.build(elements)
    print(f"Generated: {filepath}")


def build_header(title, subtitle, styles):
    return [
        Paragraph(title, styles["DocTitle"]),
        Paragraph(
            f"<b>Future Horizons Youth Foundation</b> | {subtitle}",
            styles["SubTitle"],
        ),
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#1B365D"),
            spaceAfter=15,
        ),
    ]


# ----------------------------------------------------------------------
# 1. organization_mission.pdf
# ----------------------------------------------------------------------
def gen_organization_mission():
    styles = get_styles()
    story = build_header(
        "Organization Mission & Vision",
        "Official Charter Statement",
        styles,
    )

    story.extend(
        [
            Paragraph("Our Mission", styles["SectionHeader"]),
            Paragraph(
                "Future Horizons Youth Foundation is dedicated to empowering underrepresented youth through comprehensive mentoring, education, and skill-building opportunities. We strive to break systemic barriers and create direct pathways to academic and career success.",
                styles["BodyCustom"],
            ),
            Paragraph("Our Vision", styles["SectionHeader"]),
            Paragraph(
                "A world where every young person, regardless of socio-economic background, possesses the confidence, resources, and community support needed to achieve their full potential.",
                styles["BodyCustom"],
            ),
            Paragraph("Core Values", styles["SectionHeader"]),
            Paragraph(
                "• <b>Equity & Inclusion:</b> Ensuring equal access to educational growth and leadership pathways.<br/>"
                "• <b>Community Integrity:</b> Building relationships grounded in trust, transparency, and accountability.<br/>"
                "• <b>Innovation in Learning:</b> Adapting programs to reflect evolving technology and workforce dynamics.<br/>"
                "• <b>Empowerment:</b> Equipping students to become proactive leaders in their communities.",
                styles["BodyCustom"],
            ),
            Paragraph("Founding History", styles["SectionHeader"]),
            Paragraph(
                "Established in 2018, Future Horizons began as a localized community tutoring initiative serving 35 students in downtown district public schools. Today, the foundation supports over 2,500 youth annually across multiple regional chapters.",
                styles["BodyCustom"],
            ),
        ]
    )
    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\organization_mission.pdf", story)


# ----------------------------------------------------------------------
# 2. youth_programs.pdf
# ----------------------------------------------------------------------
def gen_youth_programs():
    styles = get_styles()
    story = build_header(
        "Youth Development Programs Overview",
        "Program Directory 2025-2026",
        styles,
    )

    story.extend(
        [
            Paragraph("Program Portfolio", styles["SectionHeader"]),
            Paragraph(
                "Future Horizons operates three flagship initiatives engineered to support youth development at critical educational junctures.",
                styles["BodyCustom"],
            ),
        ]
    )

    data = [
        ["Program Name", "Target Group", "Focus Area", "Annual Reach"],
        ["Horizon Mentors", "Ages 10–18", "1-on-1 Leadership & Guidance", "850 Youth"],
        ["STEM Innovators", "Ages 12–18", "Robotics, Coding & Lab Science", "1,200 Youth"],
        ["College & Career Readiness", "Ages 16–19", "SAT Prep, Internships, Grants", "450 Youth"],
    ]

    t = Table(data, colWidths=[130, 90, 180, 100])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7FAFC")],
                ),
            ]
        )
    )

    story.append(t)
    story.append(Spacer(1, 15))
    story.append(Paragraph("Impact Summary", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "Over 92% of graduating high school seniors across all three programs successfully enroll in post-secondary education or secure certified vocational apprenticeships within 6 months of program completion.",
            styles["BodyCustom"],
        )
    )

    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\youth_programs.pdf", story)


# ----------------------------------------------------------------------
# 3. mentoring_program.pdf
# ----------------------------------------------------------------------
def gen_mentoring_program():
    styles = get_styles()
    story = build_header(
        "Horizon Mentors: Program Guidelines",
        "1-on-1 Youth Mentorship Framework",
        styles,
    )

    story.extend(
        [
            Paragraph("Program Framework", styles["SectionHeader"]),
            Paragraph(
                "The Horizon Mentors initiative pairs adult community volunteers with youth facing academic or social hurdles. Matches commit to a minimum of one academic year, meeting weekly for structured engagement.",
                styles["BodyCustom"],
            ),
            Paragraph("Core Objectives", styles["SectionHeader"]),
            Paragraph(
                "1. Improve academic engagement and classroom attendance.<br/>"
                "2. Develop emotional resilience and interpersonal communication skills.<br/>"
                "3. Expose mentees to diverse professional career opportunities.",
                styles["BodyCustom"],
            ),
            Paragraph("Volunteer Screening & Safety", styles["SectionHeader"]),
            Paragraph(
                "All mentor applicants undergo a rigorous multi-stage vetting process: multi-state criminal background checks, reference verifications, a 1-on-1 interview, and 12 hours of mandatory trauma-informed mentor training.",
                styles["BodyCustom"],
            ),
        ]
    )
    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\mentoring_program.pdf", story)


# ----------------------------------------------------------------------
# 4. stem_program.pdf
# ----------------------------------------------------------------------
def gen_stem_program():
    styles = get_styles()
    story = build_header(
        "STEM Innovators Initiative",
        "Curriculum & Laboratory Overview",
        styles,
    )

    story.extend(
        [
            Paragraph("Curriculum Focus", styles["SectionHeader"]),
            Paragraph(
                "STEM Innovators delivers hands-on technical instruction designed to bridge the digital divide. Classes run after school and during summer bootcamps in state-of-the-art community lab spaces.",
                styles["BodyCustom"],
            ),
            Paragraph("Key Modules", styles["SectionHeader"]),
            Paragraph(
                "• <b>Python & Data Fundamentals:</b> Introduction to logic, variables, and data analysis.<br/>"
                "• <b>Robotics Engineering:</b> Building and programming autonomous micro-controllers.<br/>"
                "• <b>Applied Environmental Science:</b> Field testing soil and water quality in local parks.<br/>"
                "• <b>Web Development:</b> HTML/CSS/JS fundamentals for community service websites.",
                styles["BodyCustom"],
            ),
            Paragraph("Equipment & Resources", styles["SectionHeader"]),
            Paragraph(
                "Participating students receive dedicated access to laptop workstations, 3D printers, and micro-robotics kits funded through regional corporate technology grants.",
                styles["BodyCustom"],
            ),
        ]
    )
    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\stem_program.pdf", story)


# ----------------------------------------------------------------------
# 5 & 6. annual_report_2024.pdf & annual_report_2025.pdf
# ----------------------------------------------------------------------
def gen_annual_report(year, revenue, expenses, net, participants):
    styles = get_styles()
    story = build_header(
        f"Annual Report {year}",
        f"Fiscal Year {year} Performance & Impact",
        styles,
    )

    story.extend(
        [
            Paragraph("Executive Summary", styles["SectionHeader"]),
            Paragraph(
                f"Fiscal Year {year} represented a period of milestone expansion for Future Horizons Youth Foundation. Through generous donor support and foundation grants, we expanded program operations and served over {participants:,} youth.",
                styles["BodyCustom"],
            ),
            Paragraph("Financial Performance", styles["SectionHeader"]),
        ]
    )

    data = [
        ["Financial Category", f"FY {year} Amount ($)"],
        ["Individual & Corporate Contributions", f"${revenue * 0.45:,.2f}"],
        ["Foundation & Government Grants", f"${revenue * 0.40:,.2f}"],
        ["Events & Special Initiatives", f"${revenue * 0.15:,.2f}"],
        ["Total Revenue", f"${revenue:,.2f}"],
        ["Programmatic Expenses", f"${expenses * 0.82:,.2f}"],
        ["Administrative & Operational", f"${expenses * 0.12:,.2f}"],
        ["Fundraising Expenses", f"${expenses * 0.06:,.2f}"],
        ["Total Expenses", f"${expenses:,.2f}"],
        ["Net Assets Remaining", f"${net:,.2f}"],
    ]

    t = Table(data, colWidths=[280, 220])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
                ("FONTNAME", (0, 8), (-1, 8), "Helvetica-Bold"),
                ("LINEBELOW", (0, 4), (-1, 4), 1, colors.HexColor("#1B365D")),
                ("LINEBELOW", (0, 8), (-1, 8), 1, colors.HexColor("#1B365D")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7FAFC")],
                ),
            ]
        )
    )

    story.append(t)
    create_pdf(rf"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\annual_report_{year}.pdf", story)


# ----------------------------------------------------------------------
# 7. grant_history.pdf
# ----------------------------------------------------------------------
def gen_grant_history():
    styles = get_styles()
    story = build_header(
        "Institutional Grant History",
        "Record of Awarded Funding (2022–2025)",
        styles,
    )

    story.append(
        Paragraph("Awarded Institutional Grants", styles["SectionHeader"])
    )

    data = [
        ["Grantor Organization", "Project Title", "Amount", "Period"],
        [
            "National Science Trust",
            "STEM Innovators Expansion",
            "$150,000",
            "2024–2025",
        ],
        [
            "City Youth Opportunity Fund",
            "Urban Mentorship Project",
            "$75,000",
            "2023–2024",
        ],
        [
            "TechForGood Foundation",
            "Hardware & Digital Literacy",
            "$50,000",
            "2024–2024",
        ],
        [
            "Community Health Alliance",
            "Youth Wellness & Resilience",
            "$30,000",
            "2022–2023",
        ],
        [
            "Apex Corporate Giving",
            "College Prep Scholarships",
            "$100,000",
            "2025–2026",
        ],
    ]

    t = Table(data, colWidths=[140, 180, 80, 100])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7FAFC")],
                ),
            ]
        )
    )

    story.append(t)
    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\grant_history.pdf", story)


# ----------------------------------------------------------------------
# 8. organization_policies.pdf
# ----------------------------------------------------------------------
def gen_organization_policies():
    styles = get_styles()
    story = build_header(
        "Organizational Governance & Policies",
        "Standard Operating Compliance",
        styles,
    )

    story.extend(
        [
            Paragraph("1. Child Protection Policy", styles["SectionHeader"]),
            Paragraph(
                "Future Horizons enforces a strict zero-tolerance policy regarding youth abuse, harassment, or exploitation. All staff and volunteers must undergo annual background re-checks and complete mandatory child safety reporting protocols.",
                styles["BodyCustom"],
            ),
            Paragraph("2. Conflict of Interest Policy", styles["SectionHeader"]),
            Paragraph(
                "Board members, officers, and employees must disclose any potential financial or personal conflicts of interest before voting on organizational contracts, grant distributions, or vendor selections.",
                styles["BodyCustom"],
            ),
            Paragraph("3. Financial Oversight & Controls", styles["SectionHeader"]),
            Paragraph(
                "All disbursements over $5,000 require dual approval from the Executive Director and Board Treasurer. Independent financial audits are conducted annually by a certified CPA firm.",
                styles["BodyCustom"],
            ),
            Paragraph(
                "4. Non-Discrimination Policy", styles["SectionHeader"]
            ),
            Paragraph(
                "Future Horizons prohibits discrimination on the basis of race, color, religion, sex, national origin, sexual orientation, gender identity, or disability in all program admissions and employment practices.",
                styles["BodyCustom"],
            ),
        ]
    )
    create_pdf(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\organization_policies.pdf", story)


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating organizational PDF documents...")
    gen_organization_mission()
    gen_youth_programs()
    gen_mentoring_program()
    gen_stem_program()
    gen_annual_report(
        2024,
        revenue=1250000.0,
        expenses=1100000.0,
        net=150000.0,
        participants=2100,
    )
    gen_annual_report(
        2025,
        revenue=1680000.0,
        expenses=1420000.0,
        net=260000.0,
        participants=2500,
    )
    gen_grant_history()
    gen_organization_policies()
    print("All documents generated successfully in the 'documents/' folder.")