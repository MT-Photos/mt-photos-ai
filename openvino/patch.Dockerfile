FROM mtphotos/mt-photos-ai:1.2.0

COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
EXPOSE 8060

CMD [ "python3", "/app/server.py" ]
