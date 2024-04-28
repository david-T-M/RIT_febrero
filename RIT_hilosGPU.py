import pandas as pd
import numpy as np
import utils as ut # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import spacy
import mutual_info as mi
import time
from scipy.stats import wasserstein_distance
import sys

import conceptnet_lite
conceptnet_lite.connect("../OPENAI/data/conceptnet.db")
from conceptnet_lite import Label, edges_for, edges_between

import textacy
from textacy import extract
from functools import partial


from numba import jit, cuda 

#Hilos
from threading import Thread
## esta función revisa hiperonimia, sinonimia entre otras.

def compatibilidad_semantica(word_t,word_h):
    #Nos quedamos con el lemma para identificar las relaciones blue{ blue ,ADJ}​
    wt=str(word_t).split("{")[1].split(",")[0]
    wh=str(word_h).split("{")[1].split(",")[0]
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        #Sinonimia
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="synonym"):
            print(wh," synonym ",wt)
            return True,1,"synonym"
        #Antonimos
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="antonym"):
            print(wt," antonym ",wh)
            return True,2,"antonym"
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="distinct_from"):
            print(wh," distinct from ",wt)
            return True,2,"distinct_from"
        #Hiperonimia o Hiponimia
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="is_a"):
            if wt== e.start.text:
                print(wt," is_a ",wh)
                return True,1,"is_a"
            else:
                print(wt," is_a (ant) ",wh)
                return True,2,"is_a_c"
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="manner_of"):
            if wt== e.start.text:
                print(wt," manner_of ",wh)
                return True,1,"manner_of"
            else:
                print(wt," manner_of (ant) ",wh)
                return True,2,"manner_of_c"
        for e in edges_between(concepts_wh,concepts_wt, two_way=True,relation="has_a"):
            if wt== e.start.text:
                print(wh," has_a ",wt)
                return True,1,"has_a"
            else:
                print(wh,"has_a (ant)",wt)
                return True,2,"has_a_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="derived_from"):
            if wt== e.start.text:
                print(wt," derivado_from ",wh)
                return True,1,"derived_from"
            else:
                print(wh," derivado_from (ant)",wt)
                return True,2,"derived_from_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="used_for"):
            if wt== e.start.text:
                print(wt," used_for ",wh)
                return True,1,"used_for"
            else:
                print(wh,"used_for (ant)",wt)
                return True,2,"used_for_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_property"):
            if wt== e.start.text:
                print(wt," HasProperty ",wh)
                return True,1,"HasProperty"
            else:
                print(wh," HasProperty ",wt)
                return True,2,"hasProperty_c"
        # Inflexión
        for e in edges_between(concepts_wt, concepts_wh, two_way=True,relation="form_of"):
            print(wt," form_of ",wh)
            return True,1,"form_of"
        #Implicación
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="causes"):
            if wt== e.start.text:
                print(wt," causes ",wh)
                return True,1,"Causes"
            else:
                print(wh," causes ",wt)
                return True,2,"Causes_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_prerequisite"):
            if wt== e.start.text:
                print(wt," has_prerequisite ",wh)
                return True,1,"has_prerequisite"
            else:
                print(wh," has_prerequisite ",wt)
                return True,2,"has_prerequisite_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_subevent"):
            if wt== e.start.text:
                print(wt," has_subevent ",wh)
                return True,1,"has_subevent"
            else:
                print(wh," has_subevent ",wt)
                return True,2,"has_subevent_c"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="has_first_subevent"):
            if wt== e.start.text:
                print(wt," has_first_subevent ",wh)
                return True,1,"has_first_subevent"
            else:
                print(wh," has_first_subevent ",wt)
                return True,2,"has_first_subevent_c"
            
        #relaciones conceptuales
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="related_to"):
            print(wt," related_to ",wh)
            return True,3,"related_to"
        for e in edges_between(concepts_wt,concepts_wh, two_way=True,relation="similar_to"):
            print(wt," similar_to ",wh)
            return True,3,"similar_to"
            
    except:
        pass
    return False,0,""

def obtener_distancia(texto_v,hipotesis_v,texto_t,texto_h,b_col,b_index):
    lista_l=[]
    for i in range(len(texto_t)):
        lista=[]
        for j in range(len(texto_h)):
            lista.append(np.linalg.norm(texto_v[i] - hipotesis_v[j]))#*wasserstein_distance(texto_2[i],hipotesis_2[j]))
        lista_l.append(lista)
    df_distEuc=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    df_distEuc=df_distEuc.drop(b_col[1:],axis=1)
    df_distEuc=df_distEuc.drop(b_index[1:],axis=0)
    return df_distEuc

def wasserstein_mutual_inf(texto_v,hipotesis_v,texto_t,texto_h):  
    lista_l=[]
    lista_muinfor=[]   
    for i in range(len(texto_t)):
        lista=[]
        lista_mu=[]
        for j in range(len(texto_h)):
            lista.append(wasserstein_distance(texto_v[i],hipotesis_v[j]))
            lista_mu.append(mi.mutual_information_2d(np.array(texto_v[i]),np.array(hipotesis_v[j])))
        lista_l.append(lista)
        lista_muinfor.append(lista_mu)
    DFmearth=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    DFmutual_inf=pd.DataFrame(lista_muinfor,index=texto_t,columns=texto_h)
    return DFmearth,DFmutual_inf

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    return round(sum(-p * np.log2(p) for p in probs), 3)

def bi_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-1):
        gramsT.add((palabras[i],palabras[i+1]))
    palabras = texto2
    for i in range(len(palabras)-1):
        gramsH.add((palabras[i],palabras[i+1]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def tri_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-2):
        gramsT.add((palabras[i],palabras[i+1],palabras[i+2]))
    palabras = texto2
    for i in range(len(palabras)-2):
        gramsH.add((palabras[i],palabras[i+1],palabras[i+2]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def cuatri_grams(texto1,texto2):
    gramsT=set()
    gramsH=set()
    palabras = texto1
    for i in range(len(palabras)-3):
        gramsT.add((palabras[i],palabras[i+1],palabras[i+2],palabras[i+3]))
    palabras = texto2
    for i in range(len(palabras)-3):
        gramsH.add((palabras[i],palabras[i+1],palabras[i+2],palabras[i+3]))
    if len(gramsH)!=0:
        return len(gramsT.intersection(gramsH))/len(gramsH)
    else:
        return 0

def get_grams_entities(texto):
    finales=[]
    doc = textacy.make_spacy_doc(texto, lang="en_core_web_md")
    terms = list(extract.terms(doc,
    ngs=partial(extract.ngrams, n=2, include_pos={"NOUN", "ADJ", "VERB","ADV","PART"}),
    ents=partial(extract.entities, include_types={"PERSON", "ORG", "GPE", "LOC"}),
    dedupe=False))
    a = list(extract.subject_verb_object_triples(doc))
    for e in terms:
        finales.append(str(e))
    for e in a:
        trip = ""
        for e1 in e[0]:
            trip += str(e1) + " "
        for e1 in e[1]:
            trip += str(e1) + " "
        for e1 in e[2]:
            trip += str(e1) + " "
        finales.append(trip)
    
    return finales

nlp = spacy.load("en_core_web_md") # modelo de nlp

ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv

@jit(target_backend='cuda')
def caracterizacion(url_datos,h):
    #prueba=pd.read_csv("data/DEV/pruebaDEV.csv")
    print("inicio del hilo",h)
    prueba=pd.read_csv("../OPENAI/data/"+url_datos)

    textos = prueba["sentence1"].to_list()       # almacenamiento en listas
    hipotesis = prueba["sentence2"].to_list()

    # lista de listas para dataframe
    new_data = {'sumas' : [], 'distancias' : [], 'entropia_total' : [],'entropias' : [],'mutinf' : [], 
                'mearts' : [], 'max_info' : [], 'similitud_faltantes' : [], 'list_comp' : [],
                'list_incomp' : [], 'list_rel_con' : [], 'list_M' : [], 'list_m' : [],
                'list_T' : [], 'list_relaciones' : [], 'listas_malign' : [], 'listas_malignf' : [],
                'list_bigram' : [], 'list_trigram' : [], 'list_cuatrigram' : [], 'diferencias':[],
                'bi_ent_trip_t':[],'bi_ent_trip_h':[],'bi_ent_trip_rel':[], 'clases' : []}



    inicio = time.time()
    for i in range(len(textos)):
    #for i in range(5):
        print("Hilo:",h,"Frase",i+1)

        t_vectors=ut.get_matrix_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=False,lemmatize=False)
        h_vectors=ut.get_matrix_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=False,lemmatize=False)
        t_vectors_n=ut.get_matrix_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
        h_vectors_n=ut.get_matrix_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
        t_clean=ut.get_words_rep(textos[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)
        h_clean=ut.get_words_rep(hipotesis[i],nlp,pos_to_remove=['PUNCT'], normed=True,lemmatize=False)

        # ngrams en 
        t_ngrams=get_grams_entities(textos[i])
        h_ngrams=get_grams_entities(hipotesis[i])
        print("Hilo:",h,"Frase",i+1,t_ngrams,h_ngrams)
        new_data['bi_ent_trip_t'].append(t_ngrams)
        new_data['bi_ent_trip_h'].append(h_ngrams)

        lenh=len(set(h_ngrams))
        lent=len(set(t_ngrams))

        if lenh!=0 and lent!=0:
            new_data['bi_ent_trip_rel'].append(len(set(t_ngrams).intersection(set(h_ngrams)))/lenh)
        else:
            new_data['bi_ent_trip_rel'].append(0)

        # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
        ma=np.dot(t_vectors_n,h_vectors_n.T)
        #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
        m_earth,m_mi=wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_clean,h_clean)
        ma=pd.DataFrame(ma,index=t_clean,columns=h_clean)

        # Calculamos la entropia inicial de la matriz de distancias coseno sobre tokens de T y H
        new_data['entropia_total'].append(entropia(ma.round(1).values.flatten())) 
        
        new_data['list_bigram'].append(bi_grams(t_clean,h_clean))
        new_data['list_trigram'].append(tri_grams(t_clean,h_clean))
        new_data['list_cuatrigram'].append(cuatri_grams(t_clean,h_clean))


        ###### BORRADO DE COSAS QUE NO OCUPO, SOLO NOS QUEDAMOS CON INFORMACIÓN DE TIPOS DE PALABRA: NOUN, VERB, ADJ Y ADV
        # TAMBIÉN OMITIMOS EL VERBO BE DEBIDO A QUE POR LO REGULAR SE UTILIZA COMO AUXILIAR Y ES UN VERBO COPULATIVO
        # sirve para construir la llamada predicación nominal del sujeto de una oración: 
        # #el sujeto se une con este verbo a un complemento obligatorio llamado atributo que por lo general determina 
        # alguna propiedad, estado o equivalencia del mismo, por ejemplo: "Este plato es bueno". "Juan está casado".
        
        b_col=[0]

        new_data['list_T'].append(ma.shape[0])
        new_data['list_M'].append(ma.shape[1])
        new_data['listas_malign'].append(ma)
        #print(ma.index,ma.columns)
        col=ma.columns
        borrar=[]
        indexes=ma.index
        for c in col:
            if "{null," in str(c) or "{be,VERB" in str(c) or (",NUM"  not in str(c) and "not," not in str(c) and "PRON" not in str(c) and "NOUN" not in str(c) and "VERB" not in str(c) and "ADJ" not in str(c) and "ADV" not in str(c)):
                borrar.append(c)        
            elif str(c) in indexes:
                borrar.append(c)        
        #borrar_i=[]
        #for index in indexes:
        #    if "{null," in str(index) or "{be,VERB" in str(index) or (",NUM"  not in str(index) and "not," not in str(index) and "PRON" not in str(index) and "NOUN" not in str(index) and "VERB" not in str(index) and "ADJ" not in str(index) and "ADV" not in str(index)):
        #        borrar_i.append(index) 
        #    elif str(index) in col:
        #        borrar_i.append(index) 
        
        ma=ma.drop(borrar,axis=1)
        #ma=ma.drop(borrar_i,axis=0)
        m_earth=m_earth.drop(borrar,axis=1)
        #m_earth=m_earth.drop(borrar_i,axis=0)
        m_mi=m_mi.drop(borrar,axis=1)
        #m_mi=m_mi.drop(borrar_i,axis=0)
        
        b_col.extend(borrar)
        c_compatibilidad=0
        c_incompatibilidad=0
        c_rel_concep=0

        # ELIMINAMOS INFORMACIÓN DONDE SE CORRESPONDAN EN LEMMAS, TENGA UN PRODUCTO IGUAL A 1 Y SEAN IGUALES LOS INDICES Y COLUMNAS
        # SI EL VALOR ES IGUAL A 1 SIGNIFICA QUE ES LA MISMA PALABRA, O SON SINONIMOS
        borrar=[]
        borrar_i=[]
        col=ma.columns
        for index,strings in ma.iterrows():
            lema_i=str(index).split("{")[1].split(",")[0]
            for c in col:
                if index==c:
                    borrar_i.append(index)
                    borrar.append(c)
                # if strings[c]>=1:
                #     borrar_i.append(index)
                #     borrar.append(c)
                lema_c=str(c).split("{")[1].split(",")[0]
                if lema_i == lema_c:
                    borrar_i.append(index)
                    borrar.append(c)
        ma=ma.drop(borrar,axis=1)
        #ma=ma.drop(borrar_i,axis=0)
        m_earth=m_earth.drop(borrar,axis=1)
        #m_earth=m_earth.drop(borrar_i,axis=0)
        m_mi=m_mi.drop(borrar,axis=1)
        #m_mi=m_mi.drop(borrar_i,axis=0)
        b_col.extend(borrar)
        #primera vuelta ---------------------------------------------------------------------------------
        # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
        
        n_index = ma.shape[0]
        n_columns = ma.shape[1]
        pasada=0

        parejas=""

        print("Hilo:",h,"Frase",i+1,ma.index,ma.columns)
        while n_columns>0 and pasada<3:
            borrar=[]
            a = ma.idxmax().values
            b = ma.columns
            for j in range(len(a)):
                print("Hilo:",h,"Frase",i+1,a[j],b[j])
                match,grupo,relacion = compatibilidad_semantica(a[j],b[j])
                if match and grupo==1:
                    borrar.append(b[j])
                    c_compatibilidad+=1
                    parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
                elif match and grupo==2:
                    borrar.append(b[j])
                    c_incompatibilidad+=1
                    parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
                elif match and grupo==3:
                    borrar.append(b[j])
                    c_rel_concep+=1
                    parejas = parejas + a[j] + " - " + relacion + " - "+ b[j]+ " | "
            ma = ma.drop(borrar,axis=1)
            m_earth = m_earth.drop(borrar,axis=1)
            m_mi = m_mi.drop(borrar,axis=1)
            n_columns = ma.shape[1]
            pasada+=1
            b_col.extend(borrar)
        b_index=[0]
        #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA DE CARACTERÍSTICAS
        m_distancia = obtener_distancia(t_vectors,h_vectors,t_clean,h_clean,b_col,b_index)
        
        new_data['distancias'].append(m_distancia.max().sum()) #cambie de maximas a sumas
        #m_earth=m_earth*m_distancia
        if ma.shape[1]==0:
            new_data['entropias'].append(0)
            new_data['max_info'].append(0)
            new_data['sumas'].append(0)
            new_data['mearts'].append(0)
            new_data['mutinf'].append(0)
            new_data['diferencias'].append(0)
        else:
            new_data['entropias'].append(entropia(ma.round(1).values.flatten()))
            new_data['max_info'].append(ma.max().sum()/(ma.shape[1]))# 
            new_data['sumas'].append(ma.sum().sum()/(ma.shape[1]))# 
            new_data['mearts'].append(m_earth.min().sum()/(ma.shape[1]))# 
            new_data['mutinf'].append(m_mi.max().sum()/(ma.shape[1]))# 
            new_data['diferencias'].append(len(ma.columns)/len(ma.index))

        new_data['list_comp'].append(c_compatibilidad)
        new_data['list_incomp'].append(c_incompatibilidad)
        new_data['list_rel_con'].append(c_rel_concep)
        new_data['list_relaciones'].append(parejas)
        new_data['listas_malignf'].append(ma)
        new_data['list_m'].append(ma.shape[1])

        st=""
        sh=""
        for t__1 in ma.index:
            st=st+" "+str(t__1).split("{")[0]
        for t__2 in ma.columns:
            sh=sh+" "+str(t__2).split("{")[0]
        doc1 = nlp(st)
        doc2 = nlp(sh)

        if sh!="" and st!="":
            new_data['similitud_faltantes'].append(doc1.similarity(doc2))
        elif st!="" and sh=="":
            new_data['similitud_faltantes'].append(1)
        else:
            new_data['similitud_faltantes'].append(0)
        new_data['clases'].append(prueba.at[i,"gold_label"])
    fin = time.time()

    #clases=prueba["gold_label"].values

    #temp1 =[sumas,distancias,entropia_total,entropias,mutinf,mearts,max_info,similitud_faltantes,list_comp,list_incomp,list_rel_con,list_M,list_m,list_T,list_relaciones,listas_malign,listas_malignf,list_bigram,list_trigram,list_cuatrigram,clases]
    #df_resultados = pd.DataFrame(temp1,columns=["suma","distancias","entropia_total","entropias","mutual_info","m_earth","max_info_p","sim_faltantes","Compatibilidad","Incompatibilidad","Rel_conceptuales","Shape Origin","Shape Finish","Total T","Match","Ma","Maf","bigram","trigram","cuatrigram","CLASS"])
    df_resultados = pd.DataFrame(new_data)
    #df_resultados.to_csv("salida/final/"+sys.argv[1]+".csv",index=False)
    df_resultados.to_pickle("salida/final/"+url_datos+".pickle")

    print("Tiempo que se llevo el hilo:",h,round(fin-inicio,2)," segundos")

threads = list()
for i in range(int(sys.argv[2])):
    t = Thread(target=caracterizacion, args=(sys.argv[1]+str(i+1)+".csv",i+1,))
    threads.append(t)
    t.start()
