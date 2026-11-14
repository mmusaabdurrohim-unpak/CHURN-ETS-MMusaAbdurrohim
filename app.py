from pathlib import Path
from io import BytesIO
import hashlib
import pandas as pd
import streamlit as st
from modeling import prepare, train_models

st.set_page_config(page_title='Bank Customer Churn', page_icon='🏦', layout='wide')
st.title('🏦 Bank Customer Churn')
st.caption('Eksplorasi nasabah • Perbandingan model • Prediksi churn')
upload = st.sidebar.file_uploader('Unggah CSV nasabah', type=['csv'])
base = Path(__file__).parent
candidates = [base / 'data' / 'Dataset1_Customer_Churn.csv', base / 'Dataset1_Customer_Churn.csv', base / 'data' / 'Churn_Modelling-selected-columns.csv']
default_file = next((p for p in candidates if p.is_file()), None)
if upload is None and default_file is None:
    st.info('Data bawaan belum tersedia. Unggah CSV melalui panel kiri untuk mulai menggunakan aplikasi.')
    st.caption('Untuk data bawaan, simpan Dataset1_Customer_Churn.csv di folder data atau di samping app.py.')
    st.stop()
try:
    raw = upload.getvalue() if upload is not None else default_file.read_bytes()
    st.sidebar.caption('Sumber: ' + (upload.name if upload is not None else default_file.name))
    df, features = prepare(pd.read_csv(BytesIO(raw)))
    if df.empty:
        raise ValueError('CSV tidak memiliki baris data.')
except Exception as e:
    st.error(f'Data tidak dapat dibaca: {e}')
    st.stop()
identity = hashlib.sha256(raw).hexdigest()
if st.session_state.get('identity') != identity:
    st.session_state.clear()
    st.session_state.identity = identity
st.sidebar.info('Target Exited: 0 = bertahan; 1 = keluar. Identitas nasabah tidak dipakai sebagai fitur.')
tabs = st.tabs(['Ringkasan & data', 'Eksplorasi', 'Pelatihan & evaluasi', 'Prediksi'])
with tabs[0]:
    cols = st.columns(4)
    cols[0].metric('Nasabah / baris', f'{len(df):,}')
    cols[1].metric('Fitur model', len(features))
    cols[2].metric('Rata-rata usia', f'{df.Age.mean():.1f}')
    cols[3].metric('Churn aktual', f'{df.Exited.mean():.1%}' if 'Exited' in df else 'Belum tersedia')
    if 'Exited' not in df:
        st.warning('CSV ini belum memiliki Exited. Dashboard dapat digunakan; pelatihan dan prediksi menunggu data berlabel.')
    st.dataframe(df, hide_index=True)
    st.subheader('Kelengkapan data')
    st.dataframe(pd.DataFrame({'Tipe':df.dtypes.astype(str),'Nilai kosong':df.isna().sum()}))
    st.download_button('Unduh data aktif', df.to_csv(index=False).encode('utf-8-sig'), 'data_aktif.csv', 'text/csv')
with tabs[1]:
    cat = st.selectbox('Distribusi kategori', [c for c in ['Gender','Geography','NumOfProducts','HasCrCard'] if c in df])
    st.bar_chart(df[cat].value_counts().rename('Jumlah'))
    num = st.selectbox('Distribusi numerik', [c for c in features if pd.api.types.is_numeric_dtype(df[c])])
    counts = pd.cut(df[num], bins=15, duplicates='drop').value_counts(sort=False)
    counts.index = counts.index.astype(str)
    st.bar_chart(counts.rename('Jumlah'))
    if 'Exited' in df:
        st.subheader('Tingkat churn per kategori')
        st.dataframe(df.groupby(cat,dropna=False).Exited.agg(Jumlah='size',Churn_rate='mean'))
        st.caption('Churn_rate adalah proporsi nasabah keluar dalam tiap kelompok; hubungan ini tidak membuktikan sebab-akibat.')
with tabs[2]:
    st.write('Empat model: Random Forest, Gradient Boosting, Decision Tree, dan KNN. Pembagian stratified dengan random_state=42; preprocessing hanya dipelajari dari data latih.')
    test_size = st.select_slider('Porsi data uji', options=[.20,.25,.30], value=.25)
    st.caption('Versi ini tanpa SMOTE dan tanpa fitur rasio CreditScore/Age. Angka hasil akan berbeda dari notebook. Semua metrik dihitung pada data uji; ambang klasifikasi 0,50.')
    if st.button('Latih dan bandingkan model', disabled='Exited' not in df, type='primary'):
        try:
            with st.spinner('Melatih empat model…'):
                st.session_state.trained = train_models(df, features, test_size)
                st.session_state.trained_split = test_size
        except ValueError as e:
            st.error(str(e))
    if 'trained' in st.session_state:
        scores, results, ntrain, ntest = st.session_state.trained
        st.caption(f'Hasil pelatihan terakhir: {ntrain:,} baris latih, {ntest:,} baris uji. Porsi uji {st.session_state.trained_split:.0%}.')
        st.dataframe(scores.round(4), hide_index=True)
        st.info('Accuracy = ketepatan keseluruhan; Precision = ketepatan prediksi churn; Recall = churn aktual yang terdeteksi; F1 = keseimbangan precision dan recall; ROC_AUC = kemampuan membedakan dua kelas berdasarkan probabilitas.')
        selected = st.selectbox('Detail model', list(results))
        result = results[selected]
        st.write('Confusion matrix: baris = aktual, kolom = prediksi')
        st.dataframe(pd.DataFrame(result['confusion'],index=['Aktual bertahan','Aktual keluar'],columns=['Prediksi bertahan','Prediksi keluar']))
        roc_data = []
        for name, r in results.items():
            fpr,tpr,_ = r['roc']
            roc_data.append(pd.DataFrame({'False positive rate':fpr,'True positive rate':tpr,'Model':name}))
        st.line_chart(pd.concat(roc_data),x='False positive rate',y='True positive rate',color='Model')
        st.download_button('Unduh perbandingan', scores.to_csv(index=False).encode(), 'evaluasi_model.csv','text/csv')
        st.download_button('Unduh hasil data uji model terpilih', result['test'].to_csv(index=False).encode(), 'hasil_uji.csv','text/csv')
        st.caption('Tabel ini membandingkan model pada satu holdout. Untuk klaim model terbaik, gunakan validasi silang pada data latih dan satu set uji final yang belum dipakai memilih model.')
with tabs[3]:
    if 'trained' not in st.session_state:
        st.info('Unggah CSV dengan Exited dan lakukan pelatihan terlebih dahulu.')
    else:
        results = st.session_state.trained[1]
        model_name = st.selectbox('Model prediksi', list(results), key='pred_model')
        with st.form('predict'):
            values = {}
            for c in features:
                if c in ['Gender','Geography']:
                    values[c] = st.selectbox(c, sorted(df[c].dropna().unique()))
                else:
                    values[c] = st.number_input(c, value=float(df[c].median()), min_value=0.0)
            submitted = st.form_submit_button('Prediksi nasabah')
        if submitted:
            if values['Age'] <= 0:
                st.error('Usia harus lebih dari 0.')
            else:
                prob = results[model_name]['pipeline'].predict_proba(pd.DataFrame([values]))[0,1]
                st.metric('Estimasi probabilitas churn',f'{prob:.1%}')
                st.write('Prediksi: **' + ('Keluar (1)' if prob >= .5 else 'Bertahan (0)') + '** — ambang 50%.')
                st.caption('Estimasi model untuk pembelajaran, bukan kepastian perilaku nasabah. Model hilang saat sesi server berakhir; latih ulang bila diperlukan.')
