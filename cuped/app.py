import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.optimize import minimize
import altair as alt
import hvplot.pandas
import holoviews as hv
from tqdm import tqdm

st.set_page_config(
    page_title="A/B Testing App",initial_sidebar_state="expanded"
)

def generate_data(treatment_effect, size): 
    # generate y from a normal distribution
    df = pd.DataFrame({'y': np.random.normal(loc=0, scale=1, size=size)})
    # create a covariate that's corrected with y 
    df['x'] = minimize(
        lambda x: 
        abs(0.95 - pearsonr(df.y, x)[0]), 
        np.random.rand(len(df.y))).x
    # random assign rows to two groups 0 and 1 
    df['group'] = np.random.randint(0, 2, df.shape[0])
    # for treatment group add a treatment effect 
    df.loc[df["group"] == 1, 'y'] += treatment_effect
    return df

def simulation(num_round, data_size, treatment_effect=0.5): 
    result_list = []
    for i in tqdm(range(num_round)):
        df = generate_data(treatment_effect=treatment_effect, size=data_size)
        
        diff, t, p_value = mean_t_test(df, 0, 1, 'y', "Two-sided")
        cuped_diff, cuped_t, cuped_p_value = cuped_mean_t_test(df, 0, 1, 'x', 'y', "Two-sided")
        print(df['y'])
        result_list.append(pd.Series({
            'diff': diff, 
            't': t, 
            'p_value': p_value, 
            'cuped_diff': cuped_diff, 
            'cuped_t': cuped_t, 
            'cuped_p_value': cuped_p_value
        }))  
    return pd.DataFrame(result_list)

def mean_t_test(df, group_name_a, group_name_b, y_col, hypothesis):
    a = df[df['group'] == group_name_a][y_col]
    b = df[df['group']== group_name_b][y_col]
    diff = b.mean() - a.mean()
    if hypothesis == "Two-sided":
        t, p_value = stats.ttest_ind(a, b, alternative='two-sided')
        return diff, t, p_value
    else:
        t, p_value = stats.ttest_ind(a, b)
        return diff, t, p_value/2

def cuped_mean_t_test(df, group_name_a, group_name_b, x_col, y_col, hypothesis): 
    theta = df.cov()[x_col][y_col] / df.cov()[x_col][x_col]
    df['y_cuped'] = df[y_col] - theta * df[x_col]
    a = df[df.group == group_name_a]['y_cuped']
    b = df[df.group == group_name_b]['y_cuped']
    diff = b.mean() - a.mean()
    if hypothesis == "Two-sided":
        t, p_value = stats.ttest_ind(a, b, alternative='two-sided')
        return diff, t, p_value
    else:
        t, p_value = stats.ttest_ind(a, b)
        return diff, t, p_value/2

def significance(alpha, p):
    return "YES" if p < alpha else "NO"

def plot_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(color = "#61b33b")
        .encode(
            x = alt.X("Group:O", axis = alt.Axis(labelAngle=0)),
            y = alt.Y("Difference:Q", title = "Difference (delta)"),
            opacity = "Group:O",
        )
        .properties(width=500, height =500)
    )

    chart_text = chart.mark_text(
        align = "center", baseline = "middle", dy = -10, color = "black"
        ).encode(text = alt.Text("Convension:Q", format = ", .3g"))

    return st.altair_chart((chart + chart_text).interactive())

def calculate_significance(
    df, control, treatment, covariate, result, hypothesis, alpha, 
    ):
    st.session_state.diff, st.session_state.t, st.session_state.p_value = mean_t_test(
        df, control, treatment, result, st.session_state.hypothesis)
    st.session_state.cuped_diff, st.session_state.cuped_t, st.session_state.cuped_p_value = cuped_mean_t_test(
        df, control, treatment, covariate, result, st.session_state.hypothesis)
    st.session_state.significant = significance(
        st.session_state.alpha, st.session_state.p_value
    )
    st.session_state.significant_cuped = significance(
        st.session_state.alpha, st.session_state.cuped_p_value
    )



st.write(
    """
# A/B Test & Cuped Variation Reduction
Upload your experiment results to see the significance of your A/B test after using cuped variation reduction.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type = ".csv")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

ab_default = None
result_default = None
covariate_default = None

if use_example_file:
    result_df = simulation(30, 100)

    figure = (
    (
        result_df['diff'].hvplot.kde(label='Original', title='Distribution of △: CUPED vs. MEAN estimator') * 
        result_df['cuped_diff'].hvplot.kde(label='CUPED')
    ) + 
    (
        result_df['t'].hvplot.kde(label='Original', title='Distribution of △(t-statistic): CUPED vs. MEAN estimator') * 
        result_df['cuped_t'].hvplot.kde(label='CUPED')
    )
    )

    st.bokeh_chart(hv.render(figure), use_container_width=True)
    
    ab_default = ["group"]
    result_default = ["y"]
    covariate_default = ["x"]


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Data preview")
    st.dataframe(df.head())

    st.markdown("### Select columns for analysis")
    with st.form(key = "my_form"):
        ab = st.multiselect(
            "A/B column",
            options = df.columns,
            help = "Select which column refers to your A/B testing labels.",
            default = ab_default,
        )
        if ab:
            control = df[ab[0]].unique()[0]
            treatment = df[ab[0]].unique()[1]
            decide = st.radio(
                f"Is *{treatment}* Group 1?",
                options = ["Yes", "No"],
                help = "Select yes if group B is your treatment group from your test.",
            )
            if decide == "No":
                control, treatment = treatment, control

        result = st.multiselect(
            "Result column",
            options=df.columns,
            help="Select which column shows the result of the test.",
            default=result_default,
        )

        if result:
            covariate = st.multiselect(
            "Covariate column",
            options=df.columns,
            help="Select which column is relate to result.",
            default=covariate_default,
        )
            covariate_col = covariate[0]
            result_col = result[0]
            
        with st.expander("Adjust test parameters"):
            st.markdown("### Parameters")
            st.radio(
                "Hypothesis Type",
                options = ["One-sided", "Two-sided"],
                index = 0,
                key = "hypothesis",
                help = "TBD",
            )
            st.slider(
                "Significance level (alpha)",
                min_value = 0.01,
                max_value = 0.10,
                value = 0.05,
                step = 0.01,
                key = "alpha",
                help = "The probability of mistakenly rejecting the null hypothesis, if the null hypothesis is true.",
            )
        
        submit_button = st.form_submit_button(label = "Submit")

    if not ab or not result:
        st.warning("Please select at least one option for each question.")
        st.stop()

    name = (
        "Website_Result.csv" if isinstance(uploaded_file, str) else uploaded_file.name
    )
    st.write("")
    st.write("## Result for A/B test from", name)
    st.write("")
    result_list = []

    calculate_significance(
        df,
        control,
        treatment,
        covariate_col,
        result_col,
        st.session_state.hypothesis,
        st.session_state.alpha,
    )

    mcol1, mcol2 = st.columns(2)

    # Use st.metric to diplay difference in conversion rates
    with mcol1:
        st.metric(
            "P-value",
            value=f"{(st.session_state.p_value):.3g}",
            delta=f"{(st.session_state.p_value):.3g}",
        )

        
    # Display whether or not A/B test result is statistically significant
    with mcol2:
        st.metric("Significant?", value=st.session_state.significant)


    ncol1, ncol2 = st.columns(2)
    with ncol1:
        st.metric(
            "Cuped_P-value",
            value = f"{(st.session_state.cuped_p_value):.3g}",
            delta=f"{(st.session_state.cuped_p_value):.3g}",
        )

    with ncol2:
        st.metric("Significant?", value = st.session_state.significant_cuped)



    

    






    
