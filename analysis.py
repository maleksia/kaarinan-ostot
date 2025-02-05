import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from pandasql import sqldf
from datetime import datetime
pd.set_option("styler.render.max_elements", 1000000)

# Sivun asetukset ja tyylit
st.set_page_config(
    page_title="Kaarinan ostolaskudata 2023",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .main { padding: 2rem }
    .stMetric { background-color:rgb(45, 90, 180); padding: 1rem; border-radius: 0.5rem; color: white }
    .stPlot { background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) }
    </style>
    """, unsafe_allow_html=True)

# Datan latausfunktio
@st.cache_data
def load_data():
    df = pd.read_csv('data/ostolaskudata-2023.csv', delimiter=';', encoding='latin-1')
    df['Laskun summa ilman ALV'] = df['Laskun summa ilman ALV'].str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False).astype(float)
    df['Tapaht.pvm'] = pd.to_datetime(df['Tapaht.pvm'], format='%d.%m.%Y')
    return df

def create_filters(df):
    st.sidebar.header("📊 Suodattimet")
    
    # Päivämääräalueen suodatin
    default_start = datetime(2023, 1, 1).date()
    default_end = datetime(2023, 12, 31).date()
    
    date_range = st.sidebar.date_input(
        "📅 Päivämääräalue",
        [default_start, default_end],
        min_value=df['Tapaht.pvm'].min(),
        max_value=df['Tapaht.pvm'].max()
    )
    
    # Tilin suodatin
    selected_accounts = st.sidebar.multiselect(
        "Valitse kategoriat",
        options=sorted(df['Tilin nimi'].unique())
    )
    
    # Toimittajan suodatin
    selected_suppliers = st.sidebar.multiselect(
        "Valitse toimittajat",
        options=sorted(df['Toimittajan  nimi'].unique())
    )
    
    # Summa-alueen suodatin
    min_amount, max_amount = st.sidebar.slider(
        "💶 Laskun summa-alue (€)",
        min_value=float(df['Laskun summa ilman ALV'].min()),
        max_value=float(df['Laskun summa ilman ALV'].max()),
        value=(float(df['Laskun summa ilman ALV'].min()), float(df['Laskun summa ilman ALV'].max())),
        format="%.2f €"
    )
    
    # Mahdollisuus syöttää tarkat summat
    st.sidebar.write("Tai syötä tarkat summat:")
    min_amount_input = st.sidebar.number_input("Minimi summa (€)", value=min_amount, format="%.2f")
    max_amount_input = st.sidebar.number_input("Maksimi summa (€)", value=max_amount, format="%.2f")
    
    return date_range, selected_accounts, selected_suppliers, min_amount_input, max_amount_input

# Apufunktiot visualisointia varten
def plot_top_suppliers(df):
    if df.empty:
        st.warning("Ei dataa näytettäväksi.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_suppliers = df.groupby('Toimittajan  nimi')['Laskun summa ilman ALV'].sum().sort_values(ascending=True).tail(10)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_suppliers)))
    top_suppliers.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    
    ax.set_title('Top 10 toimittajaa kokonaiskulutuksen mukaan', fontsize=16, pad=20)
    ax.set_xlabel('Kokonaismäärä (€)', fontsize=12)
    ax.set_ylabel('Toimittaja', fontsize=12)
    
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    return fig

def plot_monthly_spending(df):
    if df.empty:
        st.warning("Ei dataa näytettäväksi.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_spending = df.groupby(df['Tapaht.pvm'].dt.strftime('%Y-%m'))['Laskun summa ilman ALV'].sum()
    monthly_spending.plot(kind='line', marker='o', ax=ax, color='royalblue', linewidth=2, markersize=8)
    
    ax.set_title('Kuukausittainen kulutus', fontsize=16, pad=20)
    ax.set_xlabel('Kuukausi', fontsize=12)
    ax.set_ylabel('Kokonaismäärä (€)', fontsize=12)
    
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_account_distribution(df):
    if df.empty:
        st.warning("Ei dataa näytettäväksi.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    account_spending = df.groupby('Tilin nimi')['Laskun summa ilman ALV'].sum().sort_values(ascending=False).head(10)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(account_spending)))
    account_spending.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    
    ax.set_title('Top 10 tilit kulutuksen mukaan', fontsize=16, pad=20)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()
    return fig

def plot_invoice_distribution(df):
    if df.empty:
        st.warning("Ei dataa näytettäväksi.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_filtered = df[df['Laskun summa ilman ALV'] < df['Laskun summa ilman ALV'].quantile(0.95)]
    sns.histplot(data=df_filtered, x='Laskun summa ilman ALV', bins=50, ax=ax, color='royalblue', edgecolor='black')
    
    ax.set_title('Laskujen summien jakauma\n(poislukien ylimmät 5% paremman visualisoinnin vuoksi)', fontsize=16, pad=20)
    ax.set_xlabel('Laskun summa (€)', fontsize=12)
    ax.set_ylabel('Laskujen määrä', fontsize=12)
    
    ax.tick_params(axis='both', labelsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def apply_filters(df, date_range, selected_accounts, selected_suppliers, min_amount, max_amount):
    mask = (df['Tapaht.pvm'].dt.date >= date_range[0]) & (df['Tapaht.pvm'].dt.date <= date_range[1])
    mask &= (df['Laskun summa ilman ALV'] >= min_amount) & (df['Laskun summa ilman ALV'] <= max_amount)
    
    if selected_accounts:
        mask &= df['Tilin nimi'].isin(selected_accounts)
    if selected_suppliers:
        mask &= df['Toimittajan  nimi'].isin(selected_suppliers)
    
    return df[mask]

def create_visualizations(filtered_df):
    st.subheader("📊 Top 10 toimittajaa")
    fig1 = plot_top_suppliers(filtered_df)
    if fig1:
        st.pyplot(fig1)
    
    st.subheader("📈 Kuukausittainen kulutus")
    fig2 = plot_monthly_spending(filtered_df)
    if fig2:
        st.pyplot(fig2)
    
    st.subheader("🥧 Tilien jakauma")
    fig3 = plot_account_distribution(filtered_df)
    if fig3:
        st.pyplot(fig3)
    
    st.subheader("📊 Laskujen summien jakauma")
    fig4 = plot_invoice_distribution(filtered_df)
    if fig4:
        st.pyplot(fig4)

def display_metrics(filtered_df):
    st.subheader("📊 Avainluvut")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Laskujen määrä", 
            f"{len(filtered_df):,}",
            help="Suodatettujen laskujen kokonaismäärä"
        )
        st.metric(
            "Keskimääräinen lasku", 
            f"{filtered_df['Laskun summa ilman ALV'].mean():,.2f} €",
            help="Suodatettujen laskujen keskiarvo"
        )
    
    with col2:
        st.metric(
            "Kokonaiskulut", 
            f"{filtered_df['Laskun summa ilman ALV'].sum():,.2f} €",
            help="Suodatettujen laskujen yhteissumma"
        )
        st.metric(
            "Suurin lasku", 
            f"{filtered_df['Laskun summa ilman ALV'].max():,.2f} €",
            help="Suurimman laskun summa"
        )
    
    with col3:
        st.metric(
            "Uniikit toimittajat", 
            f"{filtered_df['Toimittajan  nimi'].nunique():,}",
            help="Eri toimittajien määrä"
        )
        st.metric(
            "Pienin lasku",
            f"{filtered_df['Laskun summa ilman ALV'].min():,.2f} €",
            help="Pienimmän laskun summa"
        )

def main():
    st.title("🏢 Kaarinan ostot vuonna 2023 datan analyysiä")
    st.write("Tämä sovellus tarjoaa yksinkertaisen tavan analysoida Kaarinan ostolaskudataa vuodelta 2023.")
    st.write("Voit suodattaa dataa päivämäärän, tilin, toimittajan ja laskun summan perusteella.")
    st.write("Lisäksi voit suorittaa SQL-kyselyitä suodatetulle datalle.")
    st.write("Data on haettu osoitteesta [Kaarinan ostolaskudata 2023](https://www.avoindata.fi/data/fi/dataset/kaarinan-kaupungin-ostolaskut-2023/resource/2b98b303-7340-44ce-aba9-2245684cdc4b)")
    
    # Lataa data
    df = load_data()
    
    # Luo ja sovella suodattimet
    date_range, selected_accounts, selected_suppliers, min_amount, max_amount = create_filters(df)
    filtered_df = apply_filters(df, date_range, selected_accounts, selected_suppliers, min_amount, max_amount)
    
    # Näytä avainluvut
    display_metrics(filtered_df)
    
    # Luo visualisoinnit
    create_visualizations(filtered_df)
    
    # Lisää SQL-kyselyosio
    st.subheader("🔍 Suorita SQL-kyselyitä")
    st.write("Voit suorittaa SQL-kyselyitä suodatetulle datalle alla.")
    
    # Näytä sarakkeiden nimet
    st.write("Sarakkeiden nimet suodatetussa datassa:")
    st.write(", ".join(filtered_df.columns))
    
    # Esimerkki SQL-kyselyt
    example_queries = [
        ("SELECT * FROM filtered_df LIMIT 10", "Näytä ensimmäiset 10 riviä suodatetusta datasta."),
        ("SELECT `Toimittajan  nimi`, SUM(`Laskun summa ilman ALV`) as Yhteensä FROM filtered_df GROUP BY `Toimittajan  nimi` ORDER BY Yhteensä DESC LIMIT 10", "Näytä top 10 toimittajaa kokonaiskulutuksen mukaan."),
        ("SELECT `Tilin nimi`, COUNT(*) FROM filtered_df GROUP BY `Tilin nimi` ORDER BY `Tilin nimi` DESC LIMIT 10", "Näytä top 10 tiliä laskujen määrän mukaan."),
        ("SELECT `Tapaht.pvm`, SUM(`Laskun summa ilman ALV`) as PäivänKulutus FROM filtered_df GROUP BY `Tapaht.pvm` ORDER BY `Tapaht.pvm`", "Näytä päivittäinen kulutus."),
        ("SELECT `Toimittajan maakoodi`, COUNT(*) as LaskujenMäärä FROM filtered_df GROUP BY `Toimittajan maakoodi` ORDER BY LaskujenMäärä DESC", "Näytä laskujen määrä toimittajan maakoodin mukaan."),
        ("SELECT CASE WHEN `Tapaht.pvm` < '2023-01-01' THEN 'Laskujen määrä ennen 2023' WHEN `Tapaht.pvm` >= '2023-01-01' THEN 'Laskujen määrä vuonna 2023' END AS aikajakso, COUNT(*) AS 'Tapahtumien määrä' FROM df GROUP BY aikajakso", "Näytä laskujen määrä koko tietokannasta ennen ja jälkeen vuoden 2023.")
    ]
    
    st.write("Esimerkki SQL-kyselyjä:")
    for query, description in example_queries:
        if st.button(description):
            st.session_state.sql_query = query
    
    # Tekstialue SQL-kyselyn syöttämistä varten
    sql_query = st.text_area("Syötä SQL-kyselysi (esim. SELECT * FROM filtered_df LIMIT 10):", value=st.session_state.get('sql_query', ''))
    
    if sql_query:
        try:
            # Suorita SQL-kysely suodatetulle DataFramelle
            result = sqldf(sql_query, locals())
            
            # Näytä tulos
            st.write("Kyselyn tulos:")
            st.dataframe(result)
        except Exception as e:
            st.error(f"Virhe SQL-kyselyn suorittamisessa: {e}")
    
    # Näytä suodatettu data sivutuksella
    if st.checkbox("Näytä raakadata"):
        # Lisää rivit per sivu valitsin
        rows_per_page = st.selectbox("Rivit per sivu", [10, 25, 50, 100], index=1)
        
        # Laske sivujen määrä
        total_rows = len(filtered_df)
        total_pages = (total_rows + rows_per_page - 1) // rows_per_page
        
        # Lisää sivuvalitsin
        if total_pages > 1:
            page = st.number_input("Sivu", min_value=1, max_value=total_pages, value=1)
        else:
            page = 1
        
        # Laske aloitus- ja lopetusindeksit
        start_idx = (page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        # Näytä sivutustiedot
        st.write(f"Näytetään rivit {start_idx + 1} - {end_idx} / {total_rows}")
        
        # Näytä sivutettu DataFrame
        st.dataframe(
            filtered_df.iloc[start_idx:end_idx].style.format({
                'Laskun summa ilman ALV': '{:,.2f} €'
            })
        )

if __name__ == "__main__":
    main()