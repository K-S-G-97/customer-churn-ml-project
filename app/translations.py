"""All user-facing text for the churn app in English, Thai and Burmese."""

TEXT = {
    "en": {
        "title": "📉 Customer Churn Risk",
        "hero_sub": "Predict which customers will leave, and decide who should get a retention offer to minimize cost.",
        "chip_recall": "🎯 93% of churners caught",
        "chip_saving": "💰 12.7% cheaper than the default",
        "chip_threshold": "⚖️ Cost-based threshold {t:.2f}",
        "theme_hint": "🌓 Light / dark: ⋮ menu → Settings → Theme",
        "tab_predict": "🔮 Predict", "tab_impact": "💰 Business impact", "tab_about": "ℹ️ About",
        "predict_header": "Predict churn risk for one customer",
        "high": "High risk", "medium": "Medium risk", "low": "Low risk",
        "prob_label": "Churn probability",
        "risk_note": "Offer threshold: {t:.0%}",
        "send_offer": "📨 Send a retention offer", "no_offer": "✅ No offer needed",
        "threshold_caption": "An offer is sent when the probability is at least {t:.2f}, the threshold that minimizes business cost.",
        "factors_title": "**Top factors for this customer** (compared with an average customer)",
        "toward_churn": "Toward churn", "toward_staying": "Toward staying",
        "push_axis": "Push on churn risk",
        "factors_caption": "Positive = pushes toward churn · Negative = pushes toward staying",
        "impact_header": "How the decision threshold affects business cost",
        "impact_caption": "Calculated on the 1,409 test-set customers the model never saw during training.",
        "offer_cost": "Offer cost ($)", "churn_cost": "Lost customer cost ($)",
        "save_rate": "Offer success rate", "threshold": "Decision threshold",
        "total_at_yours": "Total cost at your threshold", "caught": "Churners caught",
        "false_alarms": "False alarms", "cheapest": "Cheapest threshold",
        "line_total": "Total cost", "line_offer": "Offer cost", "line_churn": "Churn cost",
        "s_nothing": "Do nothing", "s_everyone": "Offer to everyone",
        "s_050": "Threshold 0.50", "s_yours": "Your threshold ({t:.2f})",
        "col_total": "Total cost",
        "pill_threshold": "THRESHOLD", "pill_recall": "RECALL", "pill_saving": "SAVING",
        "panel_risk": "CHURN RISK", "panel_factors": "TOP FACTORS",
        "panel_assumptions": "ASSUMPTIONS", "panel_curve": "COST VS THRESHOLD",
        "panel_strategies": "STRATEGY COMPARISON",
        "kpi_total": "TOTAL COST", "kpi_caught": "CHURNERS CAUGHT",
        "kpi_false": "FALSE ALARMS", "kpi_best": "CHEAPEST THRESHOLD",
        "sub_total": "At threshold 0.50: {v}", "sub_caught": "Recall: {v}",
        "sub_false": "Wasted offers: {v}", "sub_best": "Total cost there: {v}",
        "about_header": "About this project",
        "about_body": """
**Goal:** identify customers likely to churn and choose the decision threshold that minimizes business cost.

**Key finding:** at the default 0.5 threshold, the model costs *more* than offering everyone a discount.
With a cost-based threshold of **{threshold:.2f}**, it becomes the cheapest strategy, saving **12.7%** vs the
default threshold on unseen test data and catching **93%** of churners instead of 55%.

**Approach:** EDA → leakage audit → stratified split → pipeline (scaling + one-hot) →
model comparison with 5-fold CV (Logistic Regression, class weights, SMOTE, XGBoost) →
cost-based threshold → SHAP explanations.

**Assumptions:** offer cost ${offer}, lost customer ${churn}, offer saves {save:.0%} of targeted churners.
Change them in the *Business impact* tab.

**Data:** IBM Telco Customer Churn (fictional sample dataset, 7,043 customers).
""",
    },
    "th": {
        "title": "📉 ความเสี่ยงที่ลูกค้าจะเลิกใช้บริการ",
        "hero_sub": "ทำนายว่าลูกค้ารายใดจะเลิกใช้บริการ และตัดสินใจว่าใครควรได้รับข้อเสนอเพื่อให้ต้นทุนต่ำที่สุด",
        "chip_recall": "🎯 จับลูกค้าที่จะเลิกได้ 93%",
        "chip_saving": "💰 ถูกกว่าเกณฑ์มาตรฐาน 12.7%",
        "chip_threshold": "⚖️ เกณฑ์ตามต้นทุน {t:.2f}",
        "theme_hint": "🌓 โหมดสว่าง / มืด: เมนู ⋮ → Settings → Theme",
        "tab_predict": "🔮 ทำนาย", "tab_impact": "💰 ผลกระทบทางธุรกิจ", "tab_about": "ℹ️ เกี่ยวกับ",
        "predict_header": "ทำนายความเสี่ยงการเลิกใช้บริการของลูกค้าหนึ่งราย",
        "high": "ความเสี่ยงสูง", "medium": "ความเสี่ยงปานกลาง", "low": "ความเสี่ยงต่ำ",
        "prob_label": "โอกาสเลิกใช้บริการ",
        "risk_note": "เกณฑ์ส่งข้อเสนอ: {t:.0%}",
        "send_offer": "📨 ส่งข้อเสนอรักษาลูกค้า", "no_offer": "✅ ไม่จำเป็นต้องส่งข้อเสนอ",
        "threshold_caption": "จะส่งข้อเสนอเมื่อโอกาสเลิกใช้บริการอย่างน้อย {t:.2f} ซึ่งเป็นเกณฑ์ที่ทำให้ต้นทุนทางธุรกิจต่ำที่สุด",
        "factors_title": "**ปัจจัยหลักของลูกค้ารายนี้** (เทียบกับลูกค้าโดยเฉลี่ย)",
        "toward_churn": "ไปทางเลิกใช้", "toward_staying": "ไปทางใช้ต่อ",
        "push_axis": "ผลต่อความเสี่ยงการเลิกใช้",
        "factors_caption": "บวก = ผลักไปทางเลิกใช้ · ลบ = ผลักไปทางใช้ต่อ",
        "impact_header": "เกณฑ์การตัดสินใจส่งผลต่อต้นทุนทางธุรกิจอย่างไร",
        "impact_caption": "คำนวณจากลูกค้าชุดทดสอบ 1,409 รายที่โมเดลไม่เคยเห็นระหว่างการฝึก",
        "offer_cost": "ต้นทุนข้อเสนอ ($)", "churn_cost": "ต้นทุนเมื่อเสียลูกค้า ($)",
        "save_rate": "อัตราความสำเร็จของข้อเสนอ", "threshold": "เกณฑ์การตัดสินใจ",
        "total_at_yours": "ต้นทุนรวมที่เกณฑ์ของคุณ", "caught": "จับลูกค้าที่จะเลิกได้",
        "false_alarms": "แจ้งเตือนผิด", "cheapest": "เกณฑ์ที่ถูกที่สุด",
        "line_total": "ต้นทุนรวม", "line_offer": "ต้นทุนข้อเสนอ", "line_churn": "ต้นทุนการเลิกใช้",
        "s_nothing": "ไม่ทำอะไร", "s_everyone": "เสนอให้ทุกคน",
        "s_050": "เกณฑ์ 0.50", "s_yours": "เกณฑ์ของคุณ ({t:.2f})",
        "col_total": "ต้นทุนรวม",
        "pill_threshold": "เกณฑ์", "pill_recall": "อัตราจับได้", "pill_saving": "ประหยัด",
        "panel_risk": "ความเสี่ยงการเลิกใช้", "panel_factors": "ปัจจัยหลัก",
        "panel_assumptions": "สมมติฐาน", "panel_curve": "ต้นทุนเทียบกับเกณฑ์",
        "panel_strategies": "เปรียบเทียบกลยุทธ์",
        "kpi_total": "ต้นทุนรวม", "kpi_caught": "จับลูกค้าที่จะเลิกได้",
        "kpi_false": "แจ้งเตือนผิด", "kpi_best": "เกณฑ์ที่ถูกที่สุด",
        "sub_total": "ที่เกณฑ์ 0.50: {v}", "sub_caught": "อัตราการจับได้: {v}",
        "sub_false": "ข้อเสนอที่เสียเปล่า: {v}", "sub_best": "ต้นทุนรวมที่จุดนี้: {v}",
        "about_header": "เกี่ยวกับโปรเจกต์นี้",
        "about_body": """
**เป้าหมาย:** ระบุลูกค้าที่มีแนวโน้มจะเลิกใช้บริการ และเลือกเกณฑ์การตัดสินใจที่ทำให้ต้นทุนทางธุรกิจต่ำที่สุด

**ข้อค้นพบหลัก:** ที่เกณฑ์มาตรฐาน 0.5 โมเดลมีต้นทุน *สูงกว่า* การให้ส่วนลดกับลูกค้าทุกคน
เมื่อใช้เกณฑ์ตามต้นทุนที่ **{threshold:.2f}** โมเดลกลายเป็นกลยุทธ์ที่ถูกที่สุด ประหยัดได้ **12.7%** เมื่อเทียบกับเกณฑ์มาตรฐาน
บนข้อมูลทดสอบที่ไม่เคยเห็น และจับลูกค้าที่จะเลิกได้ **93%** แทนที่จะเป็น 55%

**แนวทาง:** EDA → ตรวจสอบ data leakage → stratified split → pipeline (scaling + one-hot) →
เปรียบเทียบโมเดลด้วย 5-fold CV (Logistic Regression, class weights, SMOTE, XGBoost) →
เกณฑ์ตามต้นทุน → อธิบายผลด้วย SHAP

**สมมติฐาน:** ต้นทุนข้อเสนอ ${offer}, ต้นทุนเมื่อเสียลูกค้า ${churn}, ข้อเสนอรักษาลูกค้าที่เป็นเป้าหมายได้ {save:.0%}
ปรับเปลี่ยนได้ในแท็บ *ผลกระทบทางธุรกิจ*

**ข้อมูล:** IBM Telco Customer Churn (ชุดข้อมูลตัวอย่างสมมติ, ลูกค้า 7,043 ราย)
""",
    },
    "my": {
        "title": "📉 ဖောက်သည် ထွက်ခွာနိုင်ခြေ",
        "hero_sub": "မည်သည့်ဖောက်သည်များ ထွက်ခွာမည်ကို ခန့်မှန်းပြီး ကုန်ကျစရိတ် အနည်းဆုံးဖြစ်အောင် မည်သူ့ကို ကမ်းလှမ်းချက်ပေးမည်ကို ဆုံးဖြတ်ပါ။",
        "chip_recall": "🎯 ထွက်ခွာသူ 93% ကို ဖမ်းမိ",
        "chip_saving": "💰 ပုံသေထက် 12.7% သက်သာ",
        "chip_threshold": "⚖️ ကုန်ကျစရိတ်အခြေပြု သတ်မှတ်ချက် {t:.2f}",
        "theme_hint": "🌓 အလင်း / အမှောင်: ⋮ menu → Settings → Theme",
        "tab_predict": "🔮 ခန့်မှန်းရန်", "tab_impact": "💰 စီးပွားရေး သက်ရောက်မှု", "tab_about": "ℹ️ အကြောင်း",
        "predict_header": "ဖောက်သည်တစ်ဦး၏ ထွက်ခွာနိုင်ခြေကို ခန့်မှန်းရန်",
        "high": "အန္တရာယ်မြင့်", "medium": "အန္တရာယ်အလယ်အလတ်", "low": "အန္တရာယ်နည်း",
        "prob_label": "ထွက်ခွာနိုင်ခြေ",
        "risk_note": "ကမ်းလှမ်းချက် သတ်မှတ်ချက်: {t:.0%}",
        "send_offer": "📨 ဆက်လက်ထိန်းသိမ်းရေး ကမ်းလှမ်းချက် ပို့ပါ", "no_offer": "✅ ကမ်းလှမ်းချက် မလိုအပ်ပါ",
        "threshold_caption": "ထွက်ခွာနိုင်ခြေ {t:.2f} နှင့်အထက် ဖြစ်ပါက ကမ်းလှမ်းချက် ပို့ပါမည်။ ဤသတ်မှတ်ချက်သည် စီးပွားရေးကုန်ကျစရိတ်ကို အနည်းဆုံးဖြစ်စေသည်။",
        "factors_title": "**ဤဖောက်သည်အတွက် အဓိကအကြောင်းရင်းများ** (ပျမ်းမျှဖောက်သည်နှင့် နှိုင်းယှဉ်ချက်)",
        "toward_churn": "ထွက်ခွာရန် ဦးတည်", "toward_staying": "ဆက်နေရန် ဦးတည်",
        "push_axis": "ထွက်ခွာနိုင်ခြေအပေါ် သက်ရောက်မှု",
        "factors_caption": "အပေါင်း = ထွက်ခွာရန် တွန်းအား · အနုတ် = ဆက်နေရန် တွန်းအား",
        "impact_header": "ဆုံးဖြတ်ချက် သတ်မှတ်ချက်က စီးပွားရေးကုန်ကျစရိတ်ကို မည်သို့ သက်ရောက်သနည်း",
        "impact_caption": "မော်ဒယ် လေ့ကျင့်စဉ် မမြင်ဖူးသော စမ်းသပ်ဖောက်သည် 1,409 ဦးဖြင့် တွက်ချက်ထားသည်။",
        "offer_cost": "ကမ်းလှမ်းချက် ကုန်ကျစရိတ် ($)", "churn_cost": "ဖောက်သည်ဆုံးရှုံးမှု ကုန်ကျစရိတ် ($)",
        "save_rate": "ကမ်းလှမ်းချက် အောင်မြင်နှုန်း", "threshold": "ဆုံးဖြတ်ချက် သတ်မှတ်ချက်",
        "total_at_yours": "သင့်သတ်မှတ်ချက်တွင် စုစုပေါင်းကုန်ကျစရိတ်", "caught": "ဖမ်းမိသော ထွက်ခွာသူများ",
        "false_alarms": "မှားယွင်းသော သတိပေးချက်များ", "cheapest": "အသက်သာဆုံး သတ်မှတ်ချက်",
        "line_total": "စုစုပေါင်းကုန်ကျစရိတ်", "line_offer": "ကမ်းလှမ်းချက်ကုန်ကျစရိတ်", "line_churn": "ထွက်ခွာမှုကုန်ကျစရိတ်",
        "s_nothing": "ဘာမှမလုပ်", "s_everyone": "အားလုံးကို ကမ်းလှမ်း",
        "s_050": "သတ်မှတ်ချက် 0.50", "s_yours": "သင့်သတ်မှတ်ချက် ({t:.2f})",
        "col_total": "စုစုပေါင်းကုန်ကျစရိတ်",
        "pill_threshold": "သတ်မှတ်ချက်", "pill_recall": "ဖမ်းမိနှုန်း", "pill_saving": "သက်သာမှု",
        "panel_risk": "ထွက်ခွာနိုင်ခြေ", "panel_factors": "အဓိကအကြောင်းရင်းများ",
        "panel_assumptions": "ယူဆချက်များ", "panel_curve": "ကုန်ကျစရိတ်နှင့် သတ်မှတ်ချက်",
        "panel_strategies": "နည်းဗျူဟာ နှိုင်းယှဉ်ချက်",
        "kpi_total": "စုစုပေါင်းကုန်ကျစရိတ်", "kpi_caught": "ဖမ်းမိသော ထွက်ခွာသူ",
        "kpi_false": "မှားယွင်းသတိပေးချက်", "kpi_best": "အသက်သာဆုံး သတ်မှတ်ချက်",
        "sub_total": "သတ်မှတ်ချက် 0.50 တွင်: {v}", "sub_caught": "ဖမ်းမိနှုန်း: {v}",
        "sub_false": "အလဟဿ ကမ်းလှမ်းချက်: {v}", "sub_best": "ထိုနေရာ၏ စုစုပေါင်းကုန်ကျစရိတ်: {v}",
        "about_header": "ဤပရောဂျက်အကြောင်း",
        "about_body": """
**ရည်မှန်းချက်:** ထွက်ခွာနိုင်ခြေရှိသော ဖောက်သည်များကို ဖော်ထုတ်ပြီး စီးပွားရေးကုန်ကျစရိတ် အနည်းဆုံးဖြစ်စေမည့် ဆုံးဖြတ်ချက်သတ်မှတ်ချက်ကို ရွေးချယ်ရန်။

**အဓိကတွေ့ရှိချက်:** ပုံသေ 0.5 သတ်မှတ်ချက်တွင် မော်ဒယ်သည် ဖောက်သည်အားလုံးကို လျှော့စျေးပေးခြင်းထက် *ပိုမို* ကုန်ကျသည်။
ကုန်ကျစရိတ်အခြေပြု သတ်မှတ်ချက် **{threshold:.2f}** ကို အသုံးပြုသောအခါ အသက်သာဆုံးနည်းလမ်း ဖြစ်လာပြီး
မမြင်ဖူးသော စမ်းသပ်ဒေတာတွင် ပုံသေသတ်မှတ်ချက်ထက် **12.7%** သက်သာကာ ထွက်ခွာသူ **93%** ကို ဖမ်းမိသည် (55% အစား)။

**နည်းလမ်း:** EDA → leakage စစ်ဆေးခြင်း → stratified split → pipeline (scaling + one-hot) →
5-fold CV ဖြင့် မော်ဒယ်နှိုင်းယှဉ်ခြင်း (Logistic Regression, class weights, SMOTE, XGBoost) →
ကုန်ကျစရိတ်အခြေပြု သတ်မှတ်ချက် → SHAP ရှင်းလင်းချက်။

**ယူဆချက်များ:** ကမ်းလှမ်းချက် ${offer}၊ ဖောက်သည်ဆုံးရှုံးမှု ${churn}၊ ကမ်းလှမ်းချက်သည် ပစ်မှတ်ထားသော ထွက်ခွာသူ {save:.0%} ကို ထိန်းသိမ်းနိုင်သည်။
*စီးပွားရေး သက်ရောက်မှု* tab တွင် ပြောင်းလဲကြည့်နိုင်သည်။

**ဒေတာ:** IBM Telco Customer Churn (နမူနာအဖြစ် ဖန်တီးထားသော ဒေတာ၊ ဖောက်သည် 7,043 ဦး)။
""",
    },
}

LABELS = {
    "en": {
        "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)",
        "SeniorCitizen": "Senior citizen", "Partner": "Has partner", "Dependents": "Has dependents",
        "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
        "InternetService": "Internet service", "OnlineSecurity": "Online security",
        "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
        "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies", "Contract": "Contract",
        "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    },
    "th": {
        "tenure": "ระยะเวลาใช้บริการ (เดือน)", "MonthlyCharges": "ค่าบริการรายเดือน ($)",
        "SeniorCitizen": "ผู้สูงอายุ", "Partner": "มีคู่ครอง", "Dependents": "มีผู้อยู่ในอุปการะ",
        "PhoneService": "บริการโทรศัพท์", "MultipleLines": "หลายหมายเลข",
        "InternetService": "บริการอินเทอร์เน็ต", "OnlineSecurity": "ความปลอดภัยออนไลน์",
        "OnlineBackup": "สำรองข้อมูลออนไลน์", "DeviceProtection": "ประกันอุปกรณ์",
        "TechSupport": "ฝ่ายสนับสนุนด้านเทคนิค", "StreamingTV": "สตรีมมิงทีวี",
        "StreamingMovies": "สตรีมมิงภาพยนตร์", "Contract": "สัญญา",
        "PaperlessBilling": "ใบแจ้งหนี้อิเล็กทรอนิกส์", "PaymentMethod": "วิธีชำระเงิน",
    },
    "my": {
        "tenure": "သုံးစွဲသည့်ကာလ (လ)", "MonthlyCharges": "လစဉ်ကျသင့်ငွေ ($)",
        "SeniorCitizen": "သက်ကြီးရွယ်အို", "Partner": "အိမ်ထောင်ဖက်ရှိ", "Dependents": "မှီခိုသူရှိ",
        "PhoneService": "ဖုန်းဝန်ဆောင်မှု", "MultipleLines": "ဖုန်းလိုင်းအများ",
        "InternetService": "အင်တာနက်ဝန်ဆောင်မှု", "OnlineSecurity": "အွန်လိုင်းလုံခြုံရေး",
        "OnlineBackup": "အွန်လိုင်းဘက်ကပ်", "DeviceProtection": "စက်ပစ္စည်းကာကွယ်မှု",
        "TechSupport": "နည်းပညာအကူအညီ", "StreamingTV": "တီဗီ စထရီးမင်း",
        "StreamingMovies": "ရုပ်ရှင် စထရီးမင်း", "Contract": "စာချုပ်",
        "PaperlessBilling": "စက္ကူမဲ့ ငွေတောင်းခံလွှာ", "PaymentMethod": "ငွေပေးချေမှုနည်းလမ်း",
    },
}

OPTION_LABELS = {
    "en": {},
    "th": {
        "Yes": "ใช่", "No": "ไม่",
        "No internet service": "ไม่มีอินเทอร์เน็ต", "No phone service": "ไม่มีโทรศัพท์",
        "DSL": "DSL", "Fiber optic": "ไฟเบอร์",
        "Month-to-month": "รายเดือน", "One year": "1 ปี", "Two year": "2 ปี",
        "Electronic check": "เช็คอิเล็กทรอนิกส์", "Mailed check": "เช็คทางไปรษณีย์",
        "Bank transfer (automatic)": "โอนผ่านธนาคาร (อัตโนมัติ)",
        "Credit card (automatic)": "บัตรเครดิต (อัตโนมัติ)",
    },
    "my": {
        "Yes": "ရှိ", "No": "မရှိ",
        "No internet service": "အင်တာနက် မရှိ", "No phone service": "ဖုန်း မရှိ",
        "DSL": "DSL", "Fiber optic": "ဖိုက်ဘာ",
        "Month-to-month": "လစဉ်", "One year": "တစ်နှစ်", "Two year": "နှစ်နှစ်",
        "Electronic check": "အီလက်ထရောနစ် ချက်လက်မှတ်", "Mailed check": "စာတိုက် ချက်လက်မှတ်",
        "Bank transfer (automatic)": "ဘဏ်လွှဲ (အလိုအလျောက်)",
        "Credit card (automatic)": "အကြွေးကတ် (အလိုအလျောက်)",
    },
}

SECTION_NAMES = {
    "en": {"customer": "👤 Customer", "account": "📄 Account & billing", "services": "📡 Services"},
    "th": {"customer": "👤 ลูกค้า", "account": "📄 บัญชีและการเรียกเก็บเงิน", "services": "📡 บริการ"},
    "my": {"customer": "👤 ဖောက်သည်", "account": "📄 အကောင့်နှင့် ငွေတောင်းခံမှု", "services": "📡 ဝန်ဆောင်မှုများ"},
}

LANGUAGES = {"English": "en", "ไทย": "th", "မြန်မာ": "my"}