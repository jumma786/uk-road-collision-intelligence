# How AI Can Predict Road Collision Severity — And Why It Matters for UK Road Safety

## I analysed 100,000+ real collision records to find what actually makes roads dangerous — and built a tool that predicts how severe a crash will be before it happens

---

In 2024, over **100,000 road collisions** were reported across the UK, resulting in more than **128,000 casualties**. Behind every statistic is a family affected, an emergency service stretched, and a community impacted.

But here's the thing — most of these collisions follow patterns. Certain roads, certain times, certain conditions produce far worse outcomes than others. What if we could see those patterns clearly, and use them to make smarter decisions about where to invest in road safety?

That's exactly what I set out to build.

---

## What I Built

The **UK Road Collision Intelligence Platform** takes raw government data and transforms it into actionable insights through three layers:

1. **An interactive dashboard** where anyone can explore collision patterns by location, time, weather, road type, and more
2. **A geographic hotspot map** that identifies the most dangerous locations across the UK
3. **A prediction tool** that estimates how severe a collision would be given specific road and environmental conditions

No coding required to use it. Just open the dashboard, apply filters, and explore.

---

## Where the Data Comes From

The UK Department for Transport publishes detailed records for every reported road collision. The 2024 dataset includes:

- **100,927 collision records** — where it happened, when, road conditions, severity outcome
- **128,272 casualty records** — the age, type, and severity for every person involved
- **188,000+ vehicle records** — vehicle type, driver age, engine size

This is real, verified data collected by police forces across the country. It's comprehensive, it's free, and until now, it's been sitting in spreadsheets.

---

## What the Data Reveals

### The most dangerous time isn't when you'd expect

Collisions are most *frequent* during the evening rush hour (4–6pm) — that's when traffic is heaviest. But **fatal** collisions peak much later, between **8pm and midnight**. Weekend nights are especially deadly.

This matters for resource planning. If you're allocating emergency services or police patrols purely based on collision volume, you're missing the window when the most serious incidents happen.

### Speed is the single biggest factor

The fatality rate rises dramatically once the speed limit exceeds 60mph. Rural single carriageways — country roads with the national speed limit — are the most dangerous road type by a wide margin.

This isn't surprising, but the data quantifies it precisely. It gives councils and transport authorities the evidence base to justify speed limit reviews on specific roads.

### Darkness and rural roads are a deadly combination

Dark conditions without street lighting produce the highest fatal collision rate of any lighting condition. When you combine darkness with a rural location, the risk compounds — this combination is the **single strongest predictor** of a fatal outcome in the entire dataset.

This has direct implications for infrastructure investment. Street lighting on high-risk rural stretches could save lives — and the data can identify exactly which stretches to prioritise.

### Weather matters less than people think

Fine, dry weather actually sees the **most collisions** — simply because more people drive in good conditions. The fatal *rate* does increase in fog and heavy rain, but the absolute numbers are smaller. The real danger isn't bad weather — it's darkness, speed, and isolation.

### Vulnerable road users face the worst outcomes

Motorcycle involvement significantly increases collision severity. Collisions involving heavy goods vehicles result in more casualties per incident. Elderly casualties face disproportionately severe outcomes. E-scooter-related collisions are a growing and concerning new category.

---

## How the Hotspot Map Works

Not every stretch of road is equally dangerous. Using geographic clustering, the platform identifies **collision hotspots** — locations where incidents cluster within a 500-metre radius.

Each hotspot is scored and ranked using four factors:

- **Fatality rate** — what proportion of collisions at this location are fatal
- **Serious injury rate** — what proportion result in serious injuries
- **Environmental danger** — how often collisions happen in dark, wet, or high-speed conditions
- **Volume** — how many collisions occur at this location

Hotspots are classified into four tiers: **Critical**, **High**, **Medium**, and **Low**. The top 100 most dangerous hotspots are plotted on an interactive map where stakeholders can zoom in, click on clusters, and see the risk profile for any location.

### Who benefits from this?

- **Local councils** can identify which roads in their jurisdiction need safety interventions
- **Police forces** can see which hotspots fall within their area and how they compare nationally
- **Transport planners** can prioritise infrastructure spending based on data, not assumptions
- **Insurance companies** can better understand geographic risk concentrations

---

## The Prediction Tool: "What If" Scenario Planning

The platform includes a severity prediction tool that answers a practical question: **given a specific set of conditions, how severe would a collision likely be?**

Users select from dropdown menus and sliders:

- **Road characteristics** — speed limit, road type, road class, urban or rural
- **Environmental conditions** — lighting, weather, road surface, time of day, day of week
- **Collision details** — number of vehicles, casualties, vehicle types involved
- **People involved** — driver age, casualty age, vehicle age

Hit "Predict" and the system returns:

- A severity prediction: **Fatal**, **Serious**, or **Slight**
- The probability for each outcome (e.g., 3% Fatal, 22% Serious, 75% Slight)
- A list of identified risk factors in plain language (e.g., "High speed zone," "Dark conditions," "Rural area")

### Why this matters for decision-makers

This isn't about predicting individual crashes — it's about **scenario modelling**. A transport authority could ask:

- *"If we reduce the speed limit on this rural road from 60mph to 40mph, how does the predicted severity change?"*
- *"If we install street lighting on this unlit stretch, what's the impact on fatal collision probability?"*
- *"Which combination of conditions produces the highest risk, and where do those conditions exist on our road network?"*

The prediction tool turns data into a planning instrument.

---

## Four AI Models Tested — One Clear Winner

I tested four different machine learning approaches to find the most reliable predictor:

| Approach | Strengths |
|----------|-----------|
| **Logistic Regression** | Simple, fast, interpretable — used as a baseline |
| **Random Forest** | Good overall accuracy, handles many variables well |
| **XGBoost** | Strong performance on imbalanced data |
| **LightGBM** | Best at detecting fatal collisions — the hardest and most important category |

**LightGBM** was the clear winner, particularly for identifying fatal collisions. This matters because only ~2% of collisions are fatal — a naive model could achieve 98% "accuracy" by simply never predicting a fatal outcome. LightGBM is specifically better at catching these rare but critical events.

The model was then fine-tuned through 50 rounds of automated optimisation to squeeze out maximum performance.

### What drives the predictions?

The top factors the model relies on:

1. **Number of casualties** — more people involved means more severe outcomes
2. **Speed limit** — higher speed limits strongly correlate with fatality
3. **Casualty age** — elderly people face significantly worse outcomes
4. **Vehicle size differences** — large engine capacity differences suggest car-vs-truck scenarios
5. **Composite danger score** — a combined measure of darkness, weather, surface, speed, and rural location
6. **Dark rural roads** — the interaction of these two factors is more dangerous than either alone

These aren't black-box outputs — they align with what road safety experts already know, which gives confidence that the model is learning real patterns, not statistical noise.

---

## The Dashboard: Insights at Your Fingertips

The platform is accessed through an interactive web dashboard with five sections:

### Overview
At-a-glance KPIs — total collisions, fatal count, serious injuries, total casualties. Plus charts showing how collisions distribute by hour, day of week, month, and speed limit. Everything updates instantly when you apply filters.

### Hotspot Map
Two interactive maps: one showing individual collisions colour-coded by severity (Fatal in red, Serious in orange, Slight in green), and one showing hotspot clusters sized by volume and coloured by risk tier.

### Conditions Analysis
Side-by-side comparisons of how fatal rates change across light conditions, weather, road surface, and overall danger score. Immediately reveals which environmental factors matter most.

### Model Performance
Transparent comparison of all four models tested, plus a visual breakdown of which features matter most for predictions.

### Predict Severity
The interactive scenario planning tool described above.

### Sidebar Filters
Every analytical page can be filtered by severity, urban/rural, month range, hour range, road class, speed limit, and weather. A counter shows how many records match your current selection.

---

## Built to Stay Current

Road conditions change. New roads are built, speed limits are updated, traffic patterns shift after major developments. A one-time analysis becomes outdated quickly.

This platform is designed to **keep working** as new data arrives:

- **Automated monitoring** detects when incoming data starts looking different from what the model was trained on — a signal that the model may need updating
- **Alert system** flags unusual prediction patterns, like a sudden spike in fatal predictions
- **Automated testing** runs the entire pipeline on every update to ensure nothing breaks
- The whole system can be **retrained in minutes** when new annual data is released by the DfT

---

## The Business Case

### For Local Authorities and Transport Bodies
- Evidence-based prioritisation of road safety interventions
- Identify which specific roads and junctions in your area are the most dangerous
- Model the impact of proposed changes (speed limits, lighting, road redesign) before spending

### For Emergency Services
- Allocate resources based on when and where the most *severe* incidents occur, not just the most *frequent*
- Identify temporal patterns for shift planning

### For Insurance Companies
- Understand geographic and environmental risk concentrations
- Improve risk assessment models with data-driven severity predictions

### For Road Safety Researchers and Campaigners
- Access to clear, visual evidence of which factors drive fatal outcomes
- Data to support policy recommendations with quantified impact

### For the Public
- Transparent access to road safety data for their local area
- Understanding of which conditions make driving most dangerous

---

## What's Next

This platform is a foundation. Future enhancements could include:

- **Plain-language explanations** for every prediction — not just "Serious," but specifically *why* the model thinks so
- **Weekly forecasting** — predicting how many collisions to expect in each region next week
- **Live data feeds** — connecting to real-time police incident reports instead of annual data releases
- **Before-and-after analysis** — measuring whether a specific road intervention actually reduced severity

---

## The Bottom Line

100,000 collisions a year aren't random. They follow patterns shaped by speed, darkness, road design, time of day, and location. This platform makes those patterns visible, quantifiable, and actionable.

The data is already being collected. The question is whether we use it.

---

*Data source: UK Department for Transport — Road Casualty Statistics 2024. The platform is open source and available for councils, researchers, and organisations working on road safety.*
