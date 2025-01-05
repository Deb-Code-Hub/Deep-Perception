"""
ASGI config for DeepPerception project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""
import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import HandwritingRecognition.Routing
import FaceRecognition.Routing
import ObjectDetection.Routing
import SignLanguageDetection.Routing
import ColourDetection.Routing


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DeepPerception.settings')
#application=get_asgi_application()
application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket':AuthMiddlewareStack(URLRouter([
        *HandwritingRecognition.Routing.websocket_urlpatterns,
        *FaceRecognition.Routing.websocket_urlpatterns,
        *ObjectDetection.Routing.websocket_urlpatterns,
        *SignLanguageDetection.Routing.websocket_urlpatterns,
        *ColourDetection.Routing.websocket_urlpatterns,

    ])),
})
'''websocket': AuthMiddlewareStack(URLRouter(
        FaceRecognition.Routing.websocket_urlpatterns
    )),
    'websocket': AuthMiddlewareStack(URLRouter(
        ObjectDetection.Routing.websocket_urlpatterns
    )),
    'websocket': AuthMiddlewareStack(URLRouter(
        SignLanguageDetection.Routing.websocket_urlpatterns
    )),

})'''


