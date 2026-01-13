from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('monitor/', views.monitor, name='monitor'),
    path('api/start_interview/', views.start_interview, name='start_interview'),
    path('api/end_interview/', views.end_interview, name='end_interview'),
    path('api/process_frame/', views.process_frame, name='process_frame'),
    path('api/log_tab_switch/', views.log_tab_switch, name='log_tab_switch'),
    path('api/send_audio_activity/', views.send_audio_activity, name='send_audio_activity'),
    path('api/get_candidate_details/', views.get_candidate_details, name='get_candidate_details'),
]
