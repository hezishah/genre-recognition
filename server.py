from __future__ import unicode_literals
from genre_recognizer import GenreRecognizer
from common import GENRES
import numpy as np
import os
import json
import uuid
from random import random
import time
import tornado
import tornado.ioloop
import tornado.web
from optparse import OptionParser
import youtube_dl
import urlparse
            


STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')
UPLOADS_PATH = os.path.join(os.path.dirname(__file__), 'uploads')

genre_recognizer = None

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render(os.path.join(STATIC_PATH, 'youtube.html'))
        #self.render(os.path.join(STATIC_PATH, 'index.html'))

class PlayHandler(tornado.web.RequestHandler):

    def get(self):
        self.render(os.path.join(STATIC_PATH, 'playTube.html'))

class PlayTubeHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(STATIC_PATH, 'playTube.html'))

class UploadHandler(tornado.web.RequestHandler):

    def post(self):
        file_info = self.request.files['filearg'][0]
        file_name = file_info['filename']
        file_extension = os.path.splitext(file_name)[1]
        file_uuid = str(uuid.uuid4())
        uploaded_name = file_uuid + file_extension

        if not os.path.exists(UPLOADS_PATH):
            os.makedirs(UPLOADS_PATH)

        uploaded_path = os.path.join(UPLOADS_PATH, uploaded_name)
        with open(uploaded_path, 'w') as f:
            f.write(file_info['body'])
        (predictions, duration) = genre_recognizer.recognize(
                uploaded_path)
        genre_distributions = self.get_genre_distribution_over_time(
                predictions, duration)
        json_path = os.path.join(UPLOADS_PATH, file_uuid + '.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(genre_distributions))
        self.finish('"{}"'.format(file_uuid))

    def get_genre_distribution_over_time(self, predictions, duration):
        '''
        Turns the matrix of predictions given by a model into a dict mapping
        time in the song to a music genre distribution.
        '''
        predictions = np.reshape(predictions, np.shape(predictions)[2:])
        n_steps = np.shape(predictions)[0]
        delta_t = duration / n_steps

        def get_genre_distribution(step):
            return {genre_name: float(predictions[step, genre_index])
                    for (genre_index, genre_name) in enumerate(GENRES)}

        return [((step + 1) * delta_t, get_genre_distribution(step))
                for step in xrange(n_steps)]

class YoutubeHandler(tornado.web.RequestHandler):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    def post(self):
        url = self.request.arguments['youtubeUrl'][0]
        if not url.startswith('http'):
            return
        file_name = 'youtube'
        file_extension = '.mp3'
        file_uuid = str(uuid.uuid4())
        uploaded_name = file_uuid + file_extension

        if not os.path.exists(UPLOADS_PATH):
            os.makedirs(UPLOADS_PATH)

        uploaded_path = os.path.join(UPLOADS_PATH, uploaded_name)

        self.ydl_opts['outtmpl'] = uploaded_path
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([url])
        (predictions, duration) = genre_recognizer.recognize(
                uploaded_path)
        genre_distributions = self.get_genre_distribution_over_time(
                predictions, duration)
        json_path = os.path.join(UPLOADS_PATH, file_uuid + '.json')
        url_path = os.path.join(UPLOADS_PATH, file_uuid + '.url')
        with open(url_path, 'w') as f:
            url_data = urlparse.urlparse(url)
            query = urlparse.parse_qs(url_data.query)
            video = query["v"][0]
            f.write(video)
        with open(json_path, 'w') as f:
            f.write(json.dumps(genre_distributions))
        self.finish('"{}"'.format(file_uuid))

    def get_genre_distribution_over_time(self, predictions, duration):
        '''
        Turns the matrix of predictions given by a model into a dict mapping
        time in the song to a music genre distribution.
        '''
        predictions = np.reshape(predictions, np.shape(predictions)[2:])
        n_steps = np.shape(predictions)[0]
        delta_t = duration / n_steps

        def get_genre_distribution(step):
            return {genre_name: float(predictions[step, genre_index])
                    for (genre_index, genre_name) in enumerate(GENRES)}

        return [((step + 1) * delta_t, get_genre_distribution(step))
                for step in xrange(n_steps)]

application = tornado.web.Application([
    (r'/', MainHandler),
    (r'/play.html', PlayHandler),
    (r'/static/(.*)', tornado.web.StaticFileHandler, {
        'path': STATIC_PATH
    }),
    (r'/uploads/(.*)', tornado.web.StaticFileHandler, {
        'path': UPLOADS_PATH
    }),
    (r'/upload', UploadHandler),
    (r'/youtube', YoutubeHandler),
    (r'/playTube.html', PlayTubeHandler),
], debug=True)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model_path',
            default=os.path.join(os.path.dirname(__file__), 
                'models/model.yaml'),
            help='load keras model from MODEL yaml file', metavar='MODEL')
    parser.add_option('-w', '--weights', dest='weights_path',
            default=os.path.join(os.path.dirname(__file__), 
                        'models/weights.h5'),
            help='load keras model WEIGHTS hdf5 file', metavar='WEIGHTS')
    parser.add_option('-p', '--port', dest='port',
            default=8080,
            help='run server at PORT', metavar='PORT')
    options, args = parser.parse_args()
    genre_recognizer = GenreRecognizer(options.model_path,
            options.weights_path)
    application.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
