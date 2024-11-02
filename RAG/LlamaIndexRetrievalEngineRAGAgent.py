import asyncio
from typing import Annotated
import re
import os
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, tokenize, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatImage,
)
from livekit.plugins import deepgram, openai, silero
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import MetadataMode
import requests

load_dotenv(dotenv_path=".env.local")

# Initialize RAG components
PERSIST_DIR = "./dental-knowledge-storage"
if not os.path.exists(PERSIST_DIR):
    # Load dental knowledge documents and create index
    documents = SimpleDirectoryReader("dental_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing dental knowledge index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

class DentalAssistantFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate dental issues using vision capabilities,"
            "for example, an image of teeth, gums, or the webcam feed showing the same."
        )
    )
    async def analyze_dental_image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            )
        ],
    ):
        print(f"Analyzing dental image: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description="Called when a user wants to book an appointment. This function sends a booking link to the provided email address and name."
    )
    async def book_appointment(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The email address to send the booking link to"
            ),
        ],
        name: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The name of the person booking the appointment"
            ),
        ],
    ):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "The email address seems incorrect. Please provide a valid one."

        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            headers = {'Content-Type': 'application/json'}
            data = {'email': email, 'name': name}
            response = requests.post(webhook_url, json=data, headers=headers)
            response.raise_for_status()
            return f"Dental appointment booking link sent to {email}. Please check your email."
        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your dental appointment. Please try again later."

    async def check_appointment_status(
        self,
        email: str,
    ):
        api_token = os.getenv('API_TOKEN')
        print("Checking dental appointment status")

        try:
            api_url = f"{os.getenv('CRM_CONTACT_LOOKUP_ENDPOINT')}?email={email}"
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            for contact in data.get('contacts', []):
                if 'livekit_appointment_booked' in contact.get('tags', []):
                    return "You have successfully booked a dental appointment."
            return "You haven't booked a dental appointment yet. Would you like assistance in scheduling one?"
        except requests.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error checking the dental appointment status."

    @agents.llm.ai_callable(
        description="Assess the urgency of a dental issue and determine if a human agent should be called."
    )
    async def assess_dental_urgency(
        self,
        symptoms: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Description of the dental symptoms or issues"
            ),
        ],
    ):
        urgent_keywords = ["severe pain", "swelling", "bleeding", "trauma", "knocked out", "broken"]
        if any(keyword in symptoms.lower() for keyword in urgent_keywords):
            return "call_human_agent"
        else:
            return "Your dental issue doesn't appear to be immediately urgent, but it's still important to schedule an appointment soon for a proper evaluation."

async def get_video_track(room: rtc.Room):
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    system_msg = llm.ChatMessage(
        role="system",
        content=(
            "Your name is Daela, a dental assistant for Knolabs Dental Agency. You are soft, caring with a bit of humour in you when responding. "
            "You offer appointment booking for dental care services, including urgent attention, routine check-ups, and long-term treatments available at prices according to needs which you cant say immediately. An onsite appointment is required. "
            "You can also analyze dental images to provide preliminary assessments, but always emphasize the need for professional in-person examination. "
            "Provide friendly, professional assistance and emphasize the importance of regular dental care. "
            "The users asking you questions could be of different age. so ask questions one by one. "
            "Any query outside of the dental service, politely reject stating your purpose. "
            "Keep your focus on try and get the patient's name and email address in sequence if not already provided while you help user. Encourage user to type email address to avoid any mistakes and reconfirm it after user provides it. "
            "Remember every patient asking for information or help is a potential lead for the business so always try to get the patient info being polite and not pushing too much but using subtle discussion approach"
            "If the care needed is not urgent, you can ask for image or ask user to show the dental area to use your vision capabilities to analyse the issue and offer assistance. "
            "Always keep your conversation engaging, short and multiple interactions even when the information you are sharing is lengthy and try to offer the in-person appointment."
        )
    )
    
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(system_msg)
    
    async def _will_synthesize_assistant_reply(
        assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext
    ):
        ctx_msg = system_msg.copy()
        user_msg = chat_ctx.messages[-1]
        retriever = index.as_retriever()
        nodes = await retriever.aretrieve(user_msg.content)
        ctx_msg.content += "\n\nContext that might help answer the user's question:"
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            ctx_msg.content += f"\n\n{node_content}"
        chat_ctx.messages[0] = ctx_msg  # the first message is the system message
        return assistant.llm.chat(chat_ctx=chat_ctx)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    latest_image: rtc.VideoFrame | None = None
    human_agent_present = False

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=DentalAssistantFunction(),
        chat_ctx=initial_ctx,
        will_synthesize_assistant_reply=_will_synthesize_assistant_reply,
    )

    chat = rtc.ChatManager(ctx.room)

    async def follow_up_appointment(email: str):
        fnc = assistant.fnc_ctx
        await asyncio.sleep(20)
        print(f"Finished waiting, checking dental appointment status for {email}")
        status = await fnc.check_appointment_status(email)
        await assistant.say(status, allow_interruptions=True)

    async def create_sip_participant(phone_number, room_name):
        print("trying to call an agent")
        LIVEKIT_URL = os.getenv('LIVEKIT_URL')
        LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
        LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
        SIP_TRUNK_ID = os.getenv('SIP_TRUNK_ID')

        livekit_api = api.LiveKitAPI(
            LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        )

        sip_trunk_id = SIP_TRUNK_ID
        await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                room_name=f"{room_name}",
                participant_identity=f"sip_{phone_number}",
                participant_name="Human Agent",
                play_ringtone=1
            )
        )
        await livekit_api.aclose()

    @ctx.room.on("participant_disconnected")
    def on_customer_agent_finished(RemoteParticipant: rtc.RemoteParticipant):
        print("Human Agent disconnected. AI Agent taking over.")
        asyncio.create_task(
            assistant.say(
                "Human Agent interaction completed. Politely ask if it was helpful and if user is happy to proceed with the in-person appointment.",
                allow_interruptions=True
            )
        )

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message and not human_agent_present:
            asyncio.create_task(
                assistant.say(msg.message, allow_interruptions=True)
            )
        elif msg.message and human_agent_present and "help me" in msg.message.lower():
            asyncio.create_task(
                assistant.say(msg.message, allow_interruptions=True)
            )
        else:
            print("No Assistance is needed as Human agent is tackling the communication")

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        nonlocal human_agent_present
        if len(called_functions) == 0:
            return
        
        function = called_functions[0]
        function_name = function.call_info.function_info.name
        print(function_name)
        
        if function_name == "assess_dental_urgency":
            result = function.result
            if result == "call_human_agent":
                print("calling an agent")
                human_agent_phone = os.getenv('HUMAN_AGENT_PHONE')
                asyncio.sleep(10)
                asyncio.create_task(create_sip_participant(human_agent_phone, ctx.room.name))
                human_agent_present = True
            else:
                asyncio.create_task(
                    assistant.say(result, allow_interruptions=True)
                )

        elif function_name == "book_appointment":
            email = function.call_info.arguments.get("email")
            if email:
                asyncio.create_task(follow_up_appointment(email))

    assistant.start(ctx.room)
    await assistant.say(
        "Hello! I'm Daela, your dental assistant at Knolabs Dental Agency. Can I know if you are the patient or you're representing the patient?",
        allow_interruptions=True
    )

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))