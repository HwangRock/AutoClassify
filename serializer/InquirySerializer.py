from rest_framework import serializers

class InquirySerializer(serializers.Serializer):
    text = serializers.CharField()
