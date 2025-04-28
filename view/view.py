from rest_framework.decorators import api_view
from rest_framework.response import Response
from serializer.InquirySerializer import InquirySerializer
from service import service

@api_view(["POST"])
def classify_inquiry(request):
    serializer = InquirySerializer(data=request.data)
    if serializer.is_valid():
        text = serializer.validated_data['text']
        response=service.classifyWithBert(text)
        return Response({"category": response[0], "coincidence":response[1]})
    else:
        return Response(serializer.errors, status=400)