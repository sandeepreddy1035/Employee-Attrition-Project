# Exploratory Data Analysis & Empirical Insights Report

## Executive Summary
This report summarizes the exploratory data analysis conducted on the **IBM HR Analytics Employee Attrition & Performance** dataset (1470 total employee records).

- **Total Employees Analyzed**: 1470
- **Retained Employees (No Attrition)**: 1233 (83.88%)
- **Departed Employees (Attrition)**: 237 (16.12%)
- **Imbalance Ratio**: Approximately 5.2 : 1 (Majority No)

---

## Important Methodological Note: Association vs. Causation
> [!IMPORTANT]
> The relationships presented in this analysis represent **statistical correlations and associations** observed within the historical dataset. They do not demonstrate direct causation. For instance, while employees working overtime exhibit a higher rate of attrition, overtime itself may be a proxy for burnout, project deadlines, or staffing shortages rather than an isolated cause. Retention policies should be designed holistically.

---

## Key Empirical Findings by Dimension

### 1. Overtime & Workload Impact
- Employees who worked **OverTime** exhibited an attrition rate of approximately **30.5%**, compared to only **10.4%** for non-overtime staff.
- Employees logging overtime leave at nearly **3x the rate** of their peers who do not work overtime.

### 2. Job Role & Department Dynamics
- **Sales Representatives** demonstrated the highest attrition rate among roles (~39.8%), followed by **Laboratory Technicians** (~23.9%) and **Human Resources** staff (~23.1%).
- In contrast, senior roles such as **Research Directors** (~2.5%) and **Managers** (~4.9%) showed high retention stability.
- By department, **Sales** experiences the highest relative turnover (~20.6%), followed by **Human Resources** (~19.0%) and **Research & Development** (~13.8%).

### 3. Compensation & Tenure
- Employees with lower **Monthly Income** have significantly higher likelihood of departure (median monthly income of leavers is ~$3,200 vs ~$4,900 for stayers).
- Attrition peaks within the first **2 years at the company**, indicating that early-career onboarding and new hire engagement are critical retention windows.

### 4. Job Satisfaction & Work-Life Balance
- Employees reporting level 1 ("Low") **Job Satisfaction** leave at a rate of **22.8%**, compared to **11.3%** for level 4 ("Very High").
- Employees rating their **Work-Life Balance** as 1 ("Bad") showed an attrition rate of **31.3%**, whereas levels 2, 3, and 4 had rates between 14% and 17%.

### 5. Age & Generational Trends
- Employees **under 25 years old** have an attrition rate exceeding **35%**, reflecting higher mobility and exploration in early careers.
- Turnover stabilizes significantly between ages 35 and 54 (~10-13%).

### 6. Commute Distance
- Employees living further from the workplace (median distance ~9 miles for leavers vs ~7 miles for stayers) display elevated attrition tendencies.

---

## Visualizations Generated
The following charts are saved in `outputs/visualizations/`:
1. `1_attrition_distribution.png`: Overall baseline class imbalance.
2. `2_attrition_by_overtime.png`: Direct comparison across overtime groups.
3. `3_attrition_by_jobsatisfaction.png`: Breakdown by satisfaction scale.
4. `4_attrition_by_jobrole.png`: Comparative ranking across 9 organizational roles.
5. `5_attrition_by_monthly_income.png`: Income distribution box & KDE plots.
6. `6_attrition_by_age_group.png`: Generational attrition rates.
7. `7_attrition_by_years_at_company.png`: Tenure histogram by attrition status.
8. `8_attrition_by_work_life_balance.png`: Work-life balance ratings vs attrition.
9. `9_correlation_heatmap.png`: Collinearity among numeric attributes.
10. `10_department_and_distance.png`: Department turnover and commute impact.
