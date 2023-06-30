from django.urls import path
from django.conf import settings
from django.conf.urls.static import static



from . import views

urlpatterns=[
    path('',views.home,name='home'),
    path('pagedetail/<int:id>',views.page_detail, name='pagedetail'),
    path('faq',views.faq_list,name='faq'),
    path('enquiry',views.enquiry,name='enquiry'),
    path('gallery',views.gallery,name='gallery'),
    path('gallerydetail/<int:id>',views.gallery_detail,name='gallery_detail'),
    path('pricing',views.pricing, name='pricing'),
    path('accounts/signup',views.signup,name='signup'),
    path('checkout/<int:plan_id>',views.checkout,name='checkout'),
    path('checkout_session/<int:plan_id>',views.checkout_session,name='checkout_session'),
    path('pay_success',views.pay_success, name='pay_success'),
    path('pay_cancel',views.pay_cancel, name='pay_cancel'),
    path('user-dashboard',views.user_dashboard,name='user_dashboard'),
    path('update-profile',views.update_profile,name='update_profile'),
    path('trainerlogin',views.trainerlogin,name='trainerlogin'),
    path('trainerlogout', views.trainerlogout,name='trainerlogout'),
    path('notifs',views.notifs,name='notifs'),
    path('get_notifs',views.get_notifs,name='get_notifs'),
    path('mark_read_notif',views.mark_read_notif, name='mark_read_notif'),
    
    
    ###### Retieval system##########
    ###########################################################
    #path('retrieval',views.retrieval, name='retrieval'),
    path('search_files', views.search_files, name='search_files'),
    path('preprocess', views.preprocess, name='preprocess'),
    path('create_index/', views.create_index, name='create_index'),
    path('preprocess_documents', views.preprocess_documents, name='preprocess_documents'),
    path('art_intel',views.art_intel, name='art_intel'),
    path('capture_face_photos', views.capture_face_photos, name='capture_face_photos'),
    path('identify/', views.identify_user, name='identify_user'),
   

]
if settings.DEBUG:
    urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

