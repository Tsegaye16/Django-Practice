import os
from django.shortcuts import render,redirect
from django.template.loader import get_template
from django.core import serializers
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from . import models
from . import forms
import stripe


from . forms import SignUp

from django.contrib.auth.forms import PasswordChangeForm
from django.contrib import messages
# Create your views here.

def home(request):
    banners=models.Banners.objects.all()
    services=models.Service.objects.all()[:3]
    gimgs=models.GalleryImage.objects.all().order_by('-id')
    #pages=models.Page.objects.all()
    return render(request,'home.html',{'banners':banners,'services':services,'gimgs':gimgs})

def page_detail(request,id):
    page=models.Page.objects.get(id=id)
    return render(request,'page.html',{'page':page})

def faq_list(request):
    faq=models.Faq.objects.all()
    return render(request,'faq.html',{'faqs':faq})
    
def enquiry(request):
    message=""
    if request.method=='POST':
        form=forms.EnquiryForm(request.POST)
        if form.is_valid():
            form.save()
            message='Data has been saved'
    form=forms.EnquiryForm
    return render(request,'enquiry.html',{'form':form,'message':message})


def gallery(request):
    gallery=models.Gallery.objects.all().order_by('-id')
    return render(request,'gallery.html',{'gallerys':gallery})

def gallery_detail(request,id):
    gallery=models.Gallery.objects.get(id=id)
    gallery_imgs=models.GalleryImage.objects.filter(gallery=gallery)

    return render(request,'gallery_imgs.html',{'gallery_imgs':gallery_imgs,'gallery':gallery})

def pricing(request):
    pricing=models.SubPlan.objects.all().order_by('price')
    distifeatures=models.SubPlanFeature.objects.all()
    return render(request,'pricing.html',{'plans':pricing, 'distifeatures':distifeatures})

def signup(request):
    msg=None
    if request.method=='POST':
        form=forms.SignUp(request.POST)
        #print("Huy")
        if form.is_valid():
            form.save()
            msg="You are successfully registered!"
            #return redirect('home')
    # else:
    #     form=forms.SignUp()
    #     msg="You are successfully Unregistered!"

    form=forms.SignUp
    return render(request,'registration/signup.html',{'form':form,'msg':msg})


def checkout(request,plan_id):
    planDetail=models.SubPlan.objects.get(pk=plan_id)
    
    return render(request,'checkout.html',{'plan':planDetail})

stripe.api_key='sk_test_Gx4mWEgHtCMr4DYMUIqfIrsz'


def checkout_session(request,plan_id):
    plan=models.SubPlan.objects.get(pk=plan_id)
    session=stripe.checkout.Session.create(
        payment_method_types=['card'],
            line_items= [{
      'price_data': {
        'currency': 'ETB',
        'product_data': {
         'name': plan.title,
        },
        'unit_amount': plan.price*100,
      },
      'quantity': 1,
    }],
     mode= 'payment',
    # These placeholder URLs will be replaced in a following step.
    success_url= 'http://127.0.0.1:8000/pay_success?session_id={CHECKOUT_SESSION_ID}',
    cancel_url= 'http://127.0.0.1:8000/pay_cancel',
    client_reference_id=plan_id
    )
    return redirect(session.url,code=303)
#from django.core.mail import EmailMessage
# from django.core.mail import send_mail
# from django.template.loader import render_to_string
# from django.core.mail import EmailMessage
# #from django.core.mail import EmailBackend




def pay_success(request):
    
    session = stripe.checkout.Session.retrieve(request.GET['session_id'])
    plan_id = session.client_reference_id
    plan = models.SubPlan.objects.get(pk=plan_id)
    user = request.user
    models.Subscription.objects.create(
        plan=plan,
        user=user,
        price=plan.price
    )
    # subject = 'Order Email'
    # html_content = render_to_string('orderemail.html', {'title': plan.title})
    # from_email = 'tsegayeabewa@gmail.com'
    # recipient_email = 'abewatsegaye16@gmail.com'

    # msg = EmailMessage(subject, html_content, from_email, [recipient_email])
    # msg.content_subtype = "html"
    # msg.send()

    return render(request, 'success.html')

    # subject='Order Email'
    # html_content=get_template('orderemail.html').render({'title':plan.title})
    # from_email='tsegayeabewa@gmail.com'
    
    # msg = EmailMessage(subject, html_content, from_email, ['abewatsegaye16@gmail.com'])
    # msg.content_subtype="html"
    # msg.send()

    # return render(request,'success.html')






def pay_cancel(request):
    return render(request,'cancel.html')

def user_dashboard(request):
    current_plan=models.Subscription.objects.all()
    return render(request,'user/dashboard.html',{'current_plan':current_plan})

def update_profile(request):
    msg=None
    if request.method=='POST':
        form=forms.ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            msg='Data has been saved'
    form=forms.ProfileForm(instance=request.user)
    return render(request,'user/update-profile.html',{'form':form,'msg':msg})


def trainerlogin(request):
    msg=''
    if request.method=='POST':
        username=request.POST.get('username')
        pwd=request.POST.get('pwd')
        trainer=models.Trainer.objects.filter(username=username,pwd=pwd).count()
        if trainer > 0:
            request.session['trainerLogin']=True
            return redirect('/trainer_dashboard')
        else:
            msg='Invalid'
    else:
        msg=''
    form=forms.TrainerLoginForm()
    return render(request, 'trainer/login.html',{'form':form,'msg':msg})


def trainerlogout(request):
    del request.session['trainerLogin']
    return render(request,'/trainerlogin')

def notifs(request):
    data=models.Notify.objects.all().order_by('-id')
    return render(request,'registration/notifs.html',{'data':data})

def get_notifs(request):
    data=models.Notify.objects.all().order_by('-id')
    notifStatus=False
    jsonData=[]
    totalUnread=0
    for d in data:
        
        try:
            notifStatusData=models.NotifUserStatus.objects.filter(user=request.user,notif=d)
            if notifStatusData.exists():
                notifStatus=True
        except models.NotifUserStatus.DoesNotExist:
            notifStatus=False
        if not notifStatus:
            totalUnread=totalUnread+1
        
        jsonData.append({
            'pk':d.id,
            'notify_detail':d.notify_detail,
            'notifStatus':notifStatus
        })
    #jsonData=serializers.serialize('json',data)
    return JsonResponse({'data':jsonData,'totalUnread':totalUnread})

#mark as raed
def mark_read_notif(request):
    notif=request.GET.get('notif')
    notif=models.Notify.objects.get(pk=notif)
    user=request.user
    models.NotifUserStatus.objects.create(notif=notif,user=user,status=True)
    return JsonResponse({'bool':True})
 



## INformation retrieval
#####################################
def preprocess(request):
    return render(request,'retrieval.html')


import os
from PyPDF2 import PdfReader

from django.http import HttpResponse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_documents(request):
    processed = False
    preprocessed_files = []
    
    if request.method == 'POST' and request.FILES.getlist('documents'):
        uploaded_files = request.FILES.getlist('documents')
        
        for uploaded_file in uploaded_files:
            preprocessed_file_path = os.path.join('F:/Django/GYM/preprocessed', uploaded_file.name.replace('.pdf', '.txt'))
            
            with open(preprocessed_file_path, 'w', encoding='utf-8') as output_file:
                reader = PdfReader(uploaded_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                stemmer = PorterStemmer()
                stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
                processed_text = ' '.join(stemmed_tokens)
                
                output_file.write(processed_text)
            
            original_file_path = os.path.join('F:/Django/GYM/original', uploaded_file.name)
            
            with open(original_file_path, 'wb') as file:
                file.write(uploaded_file.read())
            
            preprocessed_files.append(uploaded_file.name.replace('.pdf', '.txt'))
        
        processed = True
    
    return render(request, 'retrieval.html', {'processed': processed, 'preprocessed_files': preprocessed_files, 'show_success_message': processed})

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import time

def create_index(request):
    if request.method == 'POST' and request.POST.getlist('preprocessed_files'):
        selected_files = request.POST.getlist('preprocessed_files')
        success_message = "Indexed files created successfully."
        warning_message = "Invalid request."

        for file_name in selected_files:
            preprocessed_file_path = os.path.join('F:/Django/GYM/preprocessed', file_name)
            with open(preprocessed_file_path, 'r', encoding='utf-8') as file:
                indexed_content = f"Indexed content for {file_name}"
                indexed_file_path = os.path.join('F:/Django/GYM/indexed', file_name.replace('.txt', '.indexed'))
                with open(indexed_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(indexed_content)

        # Determine the appropriate message based on the request
        if len(selected_files) > 0:
            message = success_message
        else:
            message = warning_message

        # Render the template with the message
        template = loader.get_template('retrieval.html')
        context = {'message': message}
        rendered_template = template.render(context, request)
        return HttpResponse(rendered_template)

    return HttpResponse("Invalid request.")


import os

from django.shortcuts import render, HttpResponse

from operator import itemgetter

import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import pickle
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(search_word, preprocessed_text, search_method):
    if search_method == 'boolean':
        # Boolean search
        if search_word in preprocessed_text:
            return 1
        else:
            return 0

    elif search_method == 'vector_space':
        # Vector space search using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform([search_word, preprocessed_text])
        similarity_matrix = cosine_similarity(tfidf_vectors)
        similarity_score = similarity_matrix[0, 1]
        return similarity_score

    elif search_method == 'probabilistic':
        corpus = [preprocessed_text]  # List of preprocessed texts in the corpus
        tokenized_corpus = [text.split() for text in corpus]  # Tokenize the corpus

        # Initialize BM25 model
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize the search word
        tokenized_search_word = search_word.split()

        # Calculate BM25 similarity score
        similarity_score = bm25.get_scores(tokenized_search_word)[0]

        return similarity_score

    else:
        return 0 



def search_boolean(search_word):
    preprocessed_directory = 'F:/Django/GYM/preprocessed'
    search_results = []

    for file_name in os.listdir(preprocessed_directory):
        if file_name.endswith('.txt'):
            preprocessed_file_path = os.path.join(preprocessed_directory, file_name)
            try:
                with open(preprocessed_file_path, 'r', encoding='utf-8') as file:
                    preprocessed_text = file.read()
                if search_word in preprocessed_text:
                    relevance_score = calculate_similarity(search_word, preprocessed_text, 'boolean')
                    search_results.append((preprocessed_file_path, relevance_score))
            except Exception as e:
                print(f"Error processing {preprocessed_file_path}: {str(e)}")

    # Sort the search results based on relevance score in descending order
    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    return search_results



import codecs

def search_vector_space(search_word):
    preprocessed_directory = 'F:/Django/GYM/preprocessed'
    search_results = []

    for file_name in os.listdir(preprocessed_directory):
        if file_name.endswith('.txt'):
            preprocessed_file_path = os.path.join(preprocessed_directory, file_name)
            try:
                with open(preprocessed_file_path, 'r', encoding='utf-8') as file:
                    preprocessed_text = file.read()
                similarity_score = calculate_similarity(search_word, preprocessed_text, 'vector_space')
                if similarity_score > 0:
                    search_results.append((preprocessed_file_path, similarity_score))
            except Exception as e:
                print(f"Error processing {preprocessed_file_path}: {str(e)}")

    # Sort the search results based on similarity score in descending order
    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    return search_results
def search_files(request):
    if request.method == 'POST':
        search_word = request.POST.get('search_word', '')
        search_method = request.POST.get('search_method', '')
        message = ''

        if search_method == 'boolean':
            search_results = search_boolean(search_word)
            if not search_results:
                message = 'Search result not found.'

        elif search_method == 'vector_space':
            search_results = search_vector_space(search_word)
            if not search_results:
                message = 'Search result not found.'

        elif search_method == 'probabilistic':
            search_results = search_vector_space(search_word)
            if not search_results:
                message = 'Search result not found.'

        else:
            search_results = []
            message = 'Invalid search method.'

        return render(request, 'retrieval.html', {'search_results': search_results, 'message': message})

    return render(request, 'retrieval.html')








from collections import defaultdict
import os
from pdfminer.high_level import extract_text








import os
from PyPDF2 import PdfReader

from operator import itemgetter





import os
import PyPDF2

import os
from pdfminer import high_level
import os
import pickle
from rank_bm25 import BM25Okapi

import os
from pdfminer import high_level

def search_probabilistic(search_word):
    indexed_directory = 'F:\Django\GYM\indexed'
    search_results = []

    for indexed_file in os.listdir(indexed_directory):
        if indexed_file.endswith('.pkl'):
            indexed_file_path = os.path.join(indexed_directory, indexed_file)
            try:
                with open(indexed_file_path, 'rb') as file:
                    indexed_data = pickle.load(file)
                tokenized_corpus = indexed_data['tokenized_corpus']
                bm25 = indexed_data['bm25']
                tokenized_search_word = search_word.split()
                similarity_scores = bm25.get_scores(tokenized_search_word)
                if any(score > 0 for score in similarity_scores):
                    relevance_scores = list(zip(tokenized_corpus, similarity_scores))
                    search_results.extend(relevance_scores)
            except Exception as e:
                print(f"Error processing {indexed_file_path}: {str(e)}")

    # Sort the search results based on relevance score in descending order
    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    return search_results


# AI project#########################
import cv2
import os
import re
def art_intel(request):
    return render(request,'AI.html')


def capture_face_photos(request):
    if request.method == 'POST':
        label = request.POST.get('label')
        msg='Dataset generated successfully.'
        num_photos = int(request.POST.get('num_photos'))

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        output_folder = 'C:/Users/Addisu/Documents/AI-project/data'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        camera = cv2.VideoCapture(0)
        count = 0

        while count < num_photos:
            ret, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                output_path = os.path.join(output_folder, f"{label}_{count}.jpg")
                cv2.imwrite(output_path, face)
                count += 1

            cv2.imshow('Capture Face', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()
        template = loader.get_template('AI.html')
        context = {'msg': msg}
        rendered_template = template.render(context, request)
        return HttpResponse(rendered_template)
        

    return render(request, 'AI.html')


def identify_user(request):
    if request.method == 'POST':
        known_faces_dir = 'C:/Users/Addisu/Documents/AI-project/data'
        known_faces = []
        known_names = []

        for filename in os.listdir(known_faces_dir):
            image = cv2.imread(os.path.join(known_faces_dir, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            known_faces.append(gray)
            known_names.append(os.path.splitext(filename)[0])

        video_capture = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                found = False

                for i, known_face in enumerate(known_faces):
                    similarity = cv2.compareHist(cv2.calcHist([face], [0], None, [256], [0, 256]),
                                                 cv2.calcHist([known_face], [0], None, [256], [0, 256]),
                                                 cv2.HISTCMP_CORREL)
                    if similarity > 0.98:
                        name = known_names[i]
                        found = True
                        break

                if not found:
                    name = "Unknown"
                name = re.sub(r'[\d_]', '', name)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        
        return HttpResponse('Face identification completed.')

    return render(request, 'AI.html')
