from .resemblyzer import preprocess_wav, VoiceEncoder,sampling_rate
from pathlib import Path
from spectralcluster import SpectralClusterer
from collections import defaultdict 
from pydub import AudioSegment
from os import path
from pydub import AudioSegment

def convertmp3towav(filepath,filename):
    
    input_file = filepath
    output_file = "./AudioD/"+str(filename)+".wav"
  

    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")
    
    return output_file


def Diariazation(path,filename):
        src = path
        # dst = str(filename)+".wav"
        dst=path
        # if(path.split(".")[1]!='wav'):
        #     # dst=convertmp3towav(path,filename)
        #     return {"msg":"Only Wav files are Supported"}                                                      
        # sound = AudioSegment.from_mp3(src)
        # sound.export(dst, format="wav")



        audio_file_path = dst

        wav_fpath = Path(audio_file_path)

        wav = preprocess_wav(wav_fpath)
        encoder = VoiceEncoder("cpu")
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
        



        

        clusterer = SpectralClusterer(
            min_clusters=1,
            max_clusters=100,
            p_percentile=0.90,
            gaussian_blur_sigma=1)

        labels = clusterer.predict(cont_embeds)



        def create_labelling(labels,wav_splits):
            times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
            labelling = []
            start_time = 0

            for i,time in enumerate(times):
                if i>0 and labels[i]!=labels[i-1]:
                    temp = [str(labels[i-1]),start_time,time]
                    labelling.append(tuple(temp))
                    start_time = time
                if i==len(times)-1:
                    temp = [str(labels[i]),start_time,time]
                    labelling.append(tuple(temp))

            return labelling
        
        labelling = create_labelling(labels,wav_splits)


        
        dd=defaultdict(list)

        for tpl in labelling:
        # print(type(tpl))
        # print(tpl[0])
            dd[tpl[0]].append([tpl[1], tpl[2]])


        transcript_list=defaultdict(list)

        # t1 = t1 * 1000 #Works in milliseconds
        # t2 = t2 * 1000

        split_audio_path=defaultdict(list)
        for speaker in dd.keys():
            ind=0
            for duration in dd[speaker]:
                l=len(split_audio_path[speaker])
                t1=duration[0]*1000
                t2=duration[1]*1000
            
                newAudio = AudioSegment.from_wav(audio_file_path)
                newAudio = newAudio[t1:t2]

                save_path='./Audio/AudioD/user_'+str(speaker)+str(l)+'.wav'
                newAudio.export(save_path, format="wav") #Exports to a wav file in the current path.

                split_audio_path[speaker].append(save_path)
                
                ind+=1


        import speech_recognition as sr
        r = sr.Recognizer()
        
        for speaker in split_audio_path.keys():
            for path in split_audio_path[speaker]:
                with sr.WavFile(path) as source:              # use "test.wav" as the audio source
                    audio = r.record(source)                        # extract audio data from the file

                    try:
                        transcript=r.recognize(audio)
                        print("Transcription: " + transcript)   # recognize speech using Google Speech Recognition

                        transcript_list[speaker].append(transcript)
                    except LookupError:                                 # speech is unintelligible
                        print("Could not understand audio")
        return {"intervals":dd,"list":transcript_list}