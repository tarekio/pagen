"""Arabic document template generation via LLM.

Templates are markdown files with placeholder tokens ({WORDS_N}, {INT_A_B}, etc.)
generated once and reused many times by the dataset pipeline.  This module is
the ``pagen templates`` subcommand — it is *never* auto-invoked by the dataset
pipeline (template gen is a separate, manual step).

Ported from template_gen.py; LLM calls use pagen.llm instead of ollama directly.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys

from pagen.llm import LLMConfig, chat

MAX_TRIES = 3

_VALID_PH = re.compile(
    r"\{(?:WORDS_\d+|INT_\d+_\d+|FLOAT_[\d.]+_[\d.]+|DATE)\}"
)
_ANY_BRACE = re.compile(r"\{[^}]*\}")

# ---------------------------------------------------------------------------
# Built-in doc-type pool
# ---------------------------------------------------------------------------

RANDOM_POOL = [
    ("عقد عمل", "employment_contract"),
    ("خطاب ترقية", "promotion_letter"),
    ("خطاب تأديبي", "disciplinary_letter"),
    ("شهادة خبرة", "experience_letter"),
    ("خطاب عدم ممانعة", "noc_letter"),
    ("مقابلة الخروج", "exit_interview"),
    ("إشعار نهاية فترة تجريبية", "probation_notice"),
    ("استبيان رضا الموظفين", "employee_survey"),
    ("إشعار انتهاء عقد", "contract_termination_notice"),
    ("رخصة تجارية", "business_license"),
    ("رخصة بناء", "building_permit"),
    ("شهادة ميلاد", "birth_certificate"),
    ("وثيقة طلاق", "divorce_certificate"),
    ("عقد بيع عقار", "property_sale_agreement"),
    ("تقرير تقييم عقار", "property_valuation"),
    ("ضمان بنكي", "bank_guarantee"),
    ("خطاب اعتماد مستندي", "letter_of_credit"),
    ("إقرار ضريبي", "tax_return"),
    ("تقرير مراجع حسابات", "audit_report"),
    ("قائمة الدخل", "income_statement"),
    ("الميزانية العمومية", "balance_sheet"),
    ("جدول استهلاك الأصول", "depreciation_schedule"),
    ("مطالبة تأمين", "insurance_claim"),
    ("فاتورة جمركية", "customs_invoice"),
    ("شهادة مرضية", "sick_leave_certificate"),
    ("شهادة لياقة طبية", "fitness_certificate"),
    ("ملخص خروج مريض", "discharge_summary"),
    ("خطاب إحالة طبية", "medical_referral"),
    ("بطاقة تطعيم", "vaccination_card"),
    ("خطة غذائية", "diet_plan"),
    ("شهادة قيد دراسي", "enrollment_certificate"),
    ("خطة مادة دراسية", "course_syllabus"),
    ("إنذار أكاديمي", "academic_warning"),
    ("كشف درجات رسمي", "official_transcript"),
    ("أمر قضائي", "court_order"),
    ("تصريح خطي موثق", "affidavit"),
    ("اتفاقية شراكة", "partnership_agreement"),
    ("عقد توريد", "supply_contract"),
    ("عقد صيانة", "maintenance_contract"),
    ("تقرير مبيعات شهري", "monthly_sales_report"),
    ("خطة تسويقية", "marketing_plan"),
    ("تقرير زيارة عميل", "client_visit_report"),
    ("نموذج تحويل داخلي", "internal_transfer_form"),
    ("طلب توريد مواد", "material_supply_request"),
    ("تقرير إنتاج يومي", "daily_production_report"),
    ("جدول المناوبات", "shift_schedule"),
    ("نموذج طلب صيانة طارئة", "emergency_maintenance_request"),
    ("تقرير سلامة", "safety_report"),
    ("محضر تسليم مشروع", "project_handover_minutes"),
    ("خطاب شكر وتقدير", "appreciation_letter"),
    ("دعوة رسمية", "official_invitation"),
    ("برنامج تدريبي", "training_program"),
    ("تقرير متابعة أهداف", "goal_tracking_report"),
    ("خطة طوارئ", "emergency_plan"),
    ("تقرير التدقيق الداخلي", "internal_audit_report"),
    ("خطاب تزكية", "recommendation_letter"),
    ("نموذج طلب منحة", "grant_application"),
    ("اتفاقية مستوى الخدمة", "service_level_agreement"),
    ("تقرير جودة المنتج", "product_quality_report"),
    ("نموذج إقرار استلام", "acknowledgment_receipt"),
    ("تقرير حادثة عمل", "work_accident_report"),
]

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_EXAMPLES = [
    ("إيصال دفع", """\
# {WORDS_3}
**رقم الإيصال** {INT_1000_9000}
**التاريخ** {DATE}

**العميل** {WORDS_2}

| الوصف | المبلغ |
|---|---|
| {WORDS_2} | {FLOAT_5_50} |
| {WORDS_2} | {FLOAT_5_50} |
| **الإجمالي** | **{FLOAT_10_100}** |

شكرا لتعاملك معنا"""),

    ("طلب إجازة", """\
# طلب إجازة موظف

**اسم الموظف:** {WORDS_4}
**الرقم الوظيفي:** {INT_100_9999}
**القسم:** {WORDS_2}
**تاريخ تقديم الطلب:** {DATE}

### تفاصيل الإجازة:
نوع الإجازة المطلوبة: {WORDS_1}
تاريخ بدء الإجازة: {DATE}
تاريخ العودة للعمل: {DATE}
إجمالي عدد الأيام المطلوبة: {INT_1_30}

**ملاحظات أخرى:**
{WORDS_15}

اعتماد مديره المباشر: {WORDS_2}
توقيع الموظف: {WORDS_1}"""),

    ("مقال صحفي", """\
# {WORDS_6}

**{WORDS_2}** | {DATE}

{WORDS_60}

## {WORDS_4}

{WORDS_50}

{WORDS_40}

**{WORDS_3}:** {WORDS_20}"""),

    ("كشف راتب", """\
# كشف رواتب شهري

**الشركة:** {WORDS_3}
**الشهر:** {DATE}
**القسم:** {WORDS_2}

| م | اسم الموظف | الراتب الأساسي | البدلات | الخصومات | الصافي |
|---|---|---|---|---|---|
| ١ | {WORDS_3} | {FLOAT_3000_8000} | {FLOAT_200_1000} | {FLOAT_50_500} | {FLOAT_3000_9000} |
| ٢ | {WORDS_3} | {FLOAT_3000_8000} | {FLOAT_200_1000} | {FLOAT_50_500} | {FLOAT_3000_9000} |

---

**إجمالي الصافي للصرف:** {FLOAT_6000_18000} ريال"""),

    ("تقرير طبي", """\
# تقرير طبي

**تاريخ الزيارة:** {DATE}
**رقم الملف:** {INT_100000_999999}

**اسم المريض:** {WORDS_4}
**العمر:** {INT_10_80} سنة

## الشكوى الرئيسية
{WORDS_15}

## التشخيص
{WORDS_20}

## الخطة العلاجية
١. {WORDS_8} - {INT_1_3} مرات يومياً
٢. {WORDS_6} - عند اللزوم

**الطبيب المعالج:** د. {WORDS_2}"""),
]


def _sample_examples(n=2):
    return random.sample(_EXAMPLES, min(n, len(_EXAMPLES)))


def _build_prompt(doc_type: str) -> str:
    example_block = "\n\n".join(
        f"### مثال — {name}:\n```\n{content}\n```"
        for name, content in _sample_examples(2)
    )
    return f"""\
أنت تُنشئ قوالب وثائق عربية لمجموعة بيانات الكشف عن النصوص.

## العناصر النائبة المسموح بها فقط
- {{WORDS_N}} — N كلمة عربية (مثال: {{WORDS_3}}, {{WORDS_20}})
- {{INT_min_max}} — رقم صحيح في النطاق (مثال: {{INT_1000_9999}})
- {{FLOAT_min_max}} — رقم عشري في النطاق (مثال: {{FLOAT_100_5000}})
- {{DATE}} — تاريخ

## قواعد صارمة
- عربية فقط — لا أي نص إنجليزي
- markdown: # للعناوين, **نص** للخط العريض, - للقوائم, --- للفواصل الأفقية
- نطاقات أرقام منطقية تناسب نوع الوثيقة
- تتناسب مع صفحة A4 واحدة
- لا تشكيل في النص الثابت
- أعِد القالب الخام فقط — بلا شرح أو تعليق

{example_block}

---

أنشئ الآن قالباً لوثيقة: **{doc_type}**
"""


def _validate(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty output"
    if not _VALID_PH.search(text):
        return False, "no valid placeholders found"
    for m in _ANY_BRACE.finditer(text):
        if not _VALID_PH.match(m.group()):
            return False, f"invalid placeholder: {m.group()!r}"
    if len(text.splitlines()) < 3:
        return False, "template too short"
    return True, ""


def _slugify(name: str) -> str:
    ascii_only = re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))
    if ascii_only:
        return ascii_only
    return f"doc_{hashlib.md5(name.encode()).hexdigest()[:6]}"


# ---------------------------------------------------------------------------
# Generation and saving
# ---------------------------------------------------------------------------

def generate_template(doc_type: str, llm_config: LLMConfig) -> str:
    """Generate a markdown template for ``doc_type`` using the LLM."""
    prompt = _build_prompt(doc_type)
    for attempt in range(1, MAX_TRIES + 1):
        text = chat(llm_config, [{"role": "user", "content": prompt}])
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()
        ok, reason = _validate(text)
        if ok:
            return text
        print(f"  attempt {attempt} invalid ({reason}), retrying…", file=sys.stderr)
    raise RuntimeError(f"Could not generate a valid template after {MAX_TRIES} tries")


def save_template(text: str, slug: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{slug}.md")
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        i = 2
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        path = f"{base}_{i}{ext}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path


def pick_random(n: int, output_dir: str) -> list[tuple[str, str]]:
    """Return up to n (label, slug) pairs not already present in output_dir."""
    existing = (
        {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith(".md")}
        if os.path.isdir(output_dir) else set()
    )
    available = [(label, slug) for label, slug in RANDOM_POOL if slug not in existing]
    if not available:
        print("All pool entries already exist in the output directory.", file=sys.stderr)
        return []
    if n > len(available):
        print(f"Only {len(available)} unused pool entries available; generating all.", file=sys.stderr)
        n = len(available)
    return random.sample(available, n)


def run_generation(
    work: list[tuple[str, str]],
    llm_config: LLMConfig,
    output_dir: str,
) -> None:
    """Generate and save templates for the given (label, slug) work list."""
    total = len(work)
    for i, (label, slug) in enumerate(work, 1):
        print(f"[{i}/{total}] {label!r} → {slug}.md …")
        try:
            text = generate_template(label, llm_config)
            path = save_template(text, slug, output_dir)
            phs = len(_VALID_PH.findall(text))
            print(f"  saved → {path}  ({text.count(chr(10)) + 1} lines, {phs} placeholders)")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
