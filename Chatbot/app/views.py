from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from .models import FileModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from django.conf import settings
from faiss import write_index, read_index
import os

load_dotenv()


def get_pdf_text(pdf_docs):
   path=os.path.join(settings.MEDIA_ROOT,pdf_docs)
   text = ""
   pdf_reader = PdfReader(path)
   for page in pdf_reader.pages:
      text += page.extract_text()
   return text
   

def get_text_chunks(text):
   
   text_splitter = CharacterTextSplitter(
         separator="\n",
         chunk_size=1000,
         chunk_overlap=200,
         length_function=len
         )
   chunks = text_splitter.split_text(text)
   return chunks
    


def get_vectorstore(text_chunks):
    
   print("vectorstore")
   embeddings = OpenAIEmbeddings()
   #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
   vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
   print("vectorstore DOne")
   print(vectorstore)
   return vectorstore
    

def get_conversation_chain(vectorstore):

   if vectorstore is not None:
      llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

      memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
      conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
      return conversation_chain
   else:
      return None








def index(request):
   return render(request, "index.html")


def chat(request):
   
   if request.method == 'POST' :

      if request.session.get('file_name'):
         
         loaded_db=FAISS.load_local("./", OpenAIEmbeddings(), request.session.get('name_model'))
         conversation = get_conversation_chain(loaded_db)
         response =conversation({'question': str(request.POST.get("user_input"))})
         chat_history=response['chat_history']
         data={}
         data["Status"]=200
         for i, message in enumerate(chat_history):
            if i % 2 == 0:
               data["User_data"]=message.content
            else:
               data["chatbot_data"]=message.content
         
         
         return JsonResponse(data)
   else:
      return JsonResponse({"data": "File Uploading Fail", "Status":400})




def upload_file(request):

   if request.method == 'POST' :
        uploaded_file = request.FILES['file']
        
        try:
         file_uploaded=FileModel.objects.create(doc=uploaded_file)
         file_uploaded.save()
         file_names=FileModel.objects.get(pk=file_uploaded.id)
         name_model=str(file_names.doc).split("/")[1].split(".")[0]
         

         
         request.session['file_name']=str(file_names.doc)
         request.session['name_model']=name_model
         
         raw_text = get_pdf_text(request.session.get('file_name'))
         text_chunks = get_text_chunks(raw_text)
         vectorstore = get_vectorstore(text_chunks)
         vectorstore.save_local("./",name_model)
         
         
         return JsonResponse({"data": request.session['sname'], "Status":200})
        except Exception as e:
         print(e)
         return JsonResponse({"data": "File Uploading Fail", "Status":400})
   else:
      return JsonResponse({"data": "Incorrect Fle", "Status":400})

def delete_session(request):
   if request.method == 'POST' :
         if request.session.get('file_name'):
            FileModel.objects.filter(doc=request.session.get('file_name')).delete()
            path_cwd=str(os.getcwd())
            os.remove(path_cwd+"/"+request.session.get('name_model')+".faiss")
            os.remove(path_cwd+"/"+request.session.get('name_model')+".pkl")
            print(request.session)
            del request.session['file_name']
            del request.session['name_model']
            print(request.session.keys)
            return JsonResponse({ "Status":200})
         else:
            return JsonResponse({ "Status":400})


  
