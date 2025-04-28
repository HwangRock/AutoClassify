from rest_framework import serializers

class InquiryResponseSerializer(serializers.Serializer):
    category = serializers.CharField()
    confidence = serializers.FloatField()
