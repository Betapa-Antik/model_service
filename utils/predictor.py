import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

model = tf.keras.models.load_model("models/mosquito_classifier_mobilenetv2.h5")

class_names = [
'adult_aedes_aegypti','adult_aedes_albopictus','adult_aedes_canadensis',
'adult_aedes_dorsalis','adult_aedes_geniculatus','adult_aedes_koreicus',
'adult_aedes_triseriatus','adult_aedes_vexans','adult_anopheles_arabiensis',
'adult_anopheles_freeborni','adult_anopheles_sinensis','adult_class_background',
'adult_culex_inatomii','adult_culex_pipiens','adult_culex_quinquefasciatus',
'adult_culex_tritaeniorhynchus','adult_culiseta_annulata','adult_culiseta_longiareolata',
'larva_Aedes','larva_Anopheles','larva_Culex',
'not_mosquito_bicycle','not_mosquito_bird','not_mosquito_car','not_mosquito_cat',
'not_mosquito_chair','not_mosquito_dog','not_mosquito_laptop','not_mosquito_person',
'not_mosquito_tree'
]

THRESHOLD = 0.4

# ======================================================
# DESKRIPSI SPESIFIK SETIAP SPESIES NYAMUK
# ======================================================

DESCRIPTIONS = {

"Aedes aegypti":
"Aedes aegypti adalah nyamuk berukuran kecil berwarna hitam dengan pola garis putih pada kaki dan tubuhnya. "
"Ciri khasnya adalah pola seperti lyre pada bagian thorax. Nyamuk ini aktif menggigit pada pagi dan sore hari "
"dan sangat menyukai lingkungan perkotaan. Habitat berkembang biaknya adalah air bersih yang tergenang seperti "
"bak mandi, kaleng bekas, pot bunga, dan tempat penampungan air. Spesies ini merupakan vektor utama penyakit "
"demam berdarah dengue (DBD), chikungunya, dan virus Zika.",

"Aedes albopictus":
"Aedes albopictus dikenal sebagai Asian tiger mosquito karena memiliki garis putih mencolok di sepanjang "
"punggungnya dan pola belang hitam putih pada kakinya. Spesies ini dapat hidup di lingkungan hutan maupun "
"perkotaan. Habitat larvanya biasanya di genangan air kecil seperti ban bekas atau wadah alami. Nyamuk ini "
"merupakan vektor potensial penyakit dengue, chikungunya, dan Zika.",

"Aedes canadensis":
"Aedes canadensis merupakan nyamuk yang banyak ditemukan di daerah hutan di Amerika Utara. Tubuhnya berwarna "
"coklat gelap dengan pola sisik terang pada kaki dan tubuh. Nyamuk ini biasanya berkembang di genangan air "
"alami seperti rawa dan kolam kecil. Spesies ini jarang menjadi vektor penyakit pada manusia.",

"Aedes dorsalis":
"Aedes dorsalis adalah spesies nyamuk yang sering ditemukan di daerah rawa dan pesisir. Memiliki tubuh coklat "
"dengan pola sisik pucat. Nyamuk ini dapat berkembang di air dengan kadar garam tinggi dan sering ditemukan "
"di habitat payau.",

"Aedes geniculatus":
"Aedes geniculatus biasanya berkembang di lubang pohon yang berisi air hujan. Nyamuk ini berwarna gelap dengan "
"garis putih pada kaki. Habitatnya lebih sering ditemukan di daerah hutan atau taman dengan pepohonan besar.",

"Aedes koreicus":
"Aedes koreicus merupakan spesies invasif yang berasal dari Asia Timur. Nyamuk ini memiliki pola sisik terang "
"pada kaki dan thorax yang mirip dengan Aedes japonicus. Spesies ini mampu bertahan pada suhu lebih dingin "
"dibandingkan beberapa spesies Aedes lainnya.",

"Aedes triseriatus":
"Aedes triseriatus dikenal sebagai tree-hole mosquito karena larvanya berkembang di lubang pohon berisi air. "
"Nyamuk ini merupakan vektor utama virus La Crosse encephalitis di Amerika Utara.",

"Aedes vexans":
"Aedes vexans merupakan salah satu nyamuk paling umum di dunia. Nyamuk ini sering muncul dalam jumlah besar "
"setelah hujan lebat atau banjir karena berkembang biak di genangan air sementara. Nyamuk ini sangat agresif "
"dalam menggigit manusia.",

"Anopheles arabiensis":
"Anopheles arabiensis merupakan salah satu vektor utama malaria di Afrika. Ciri khas nyamuk Anopheles adalah "
"posisi tubuh saat hinggap yang membentuk sudut dengan permukaan. Sayapnya memiliki bercak-bercak gelap.",

"Anopheles freeborni":
"Anopheles freeborni adalah spesies nyamuk yang ditemukan di Amerika Utara dan dikenal sebagai vektor malaria "
"di wilayah tersebut. Nyamuk ini biasanya berkembang di perairan yang relatif bersih seperti kolam atau "
"saluran irigasi.",

"Anopheles sinensis":
"Anopheles sinensis merupakan vektor malaria yang umum ditemukan di Asia Timur dan Asia Tenggara. Nyamuk ini "
"biasanya berkembang di sawah atau perairan dangkal dengan vegetasi.",

"Culex pipiens":
"Culex pipiens dikenal sebagai nyamuk rumah atau common house mosquito. Nyamuk ini aktif pada malam hari dan "
"sering berkembang biak di air yang relatif kotor seperti selokan atau genangan air limbah. Spesies ini dapat "
"menularkan virus West Nile.",

"Culex quinquefasciatus":
"Culex quinquefasciatus adalah nyamuk yang banyak ditemukan di daerah tropis dan subtropis. Nyamuk ini "
"merupakan vektor penyakit filariasis (kaki gajah) dan virus West Nile.",

"Culex tritaeniorhynchus":
"Culex tritaeniorhynchus merupakan vektor utama virus Japanese Encephalitis. Nyamuk ini biasanya berkembang "
"di sawah dan daerah pertanian dengan genangan air.",

"Culex inatomii":
"Culex inatomii merupakan spesies nyamuk yang ditemukan di habitat perairan tertentu di Asia. Informasi tentang "
"perannya sebagai vektor penyakit masih terbatas.",

"Culiseta annulata":
"Culiseta annulata adalah nyamuk berukuran relatif besar yang banyak ditemukan di wilayah Eropa dan Asia. "
"Spesies ini biasanya berkembang di genangan air alami seperti kolam atau rawa.",

"Culiseta longiareolata":
"Culiseta longiareolata adalah nyamuk yang berkembang di genangan air yang relatif bersih seperti kolam atau "
"wadah air hujan.",

"Aedes":
"Larva Aedes memiliki posisi menggantung di permukaan air dengan sifon pernapasan yang pendek. Larva ini "
"biasanya ditemukan di air bersih yang tergenang seperti bak mandi, ember, atau wadah air.",

"Anopheles":
"Larva Anopheles memiliki posisi sejajar dengan permukaan air karena tidak memiliki sifon pernapasan panjang. "
"Larva ini sering ditemukan di perairan bersih seperti kolam atau sawah.",

"Culex":
"Larva Culex biasanya menggantung di permukaan air dengan sifon pernapasan yang panjang dan hidup di air yang "
"lebih keruh atau tercemar seperti selokan."
}


def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(image):

    img = preprocess_image(image)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    label = class_names[index]

    if confidence < THRESHOLD:
        return {
            "result": "Gambar tidak dapat dikenali",
            "confidence_percent": round(confidence * 100, 2)
        }

    if label.startswith("not_mosquito"):

        return {
            "result": "Gambar tidak bisa dianalisa karena bukan nyamuk",
            "confidence_percent": round(confidence * 100, 2)
        }

    if label.startswith("adult_"):

        species_key = label.replace("adult_", "").replace("_", " ")
        species = species_key.capitalize()

        description = DESCRIPTIONS.get(species_key, "Deskripsi tidak tersedia.")

        return {
            "type": "Nyamuk Dewasa",
            "species": species,
            "result": f"{species} (Nyamuk Dewasa)",
            "confidence_percent": round(confidence * 100, 2),
            "description": description
        }

    if label.startswith("larva_"):

        genus = label.replace("larva_", "")

        description = DESCRIPTIONS.get(genus, "Deskripsi tidak tersedia.")

        return {
            "type": "Larva Nyamuk",
            "species": genus,
            "result": f"{genus} (Larva)",
            "confidence_percent": round(confidence * 100, 2),
            "description": description
        }