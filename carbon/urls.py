from django.urls import path
from .views import fetch_and_predict_carbon

urlpatterns = [
    path('ai-carbon-estimation/<str:address>/', fetch_and_predict_carbon, name='fetch_and_predict_carbon'),
]
