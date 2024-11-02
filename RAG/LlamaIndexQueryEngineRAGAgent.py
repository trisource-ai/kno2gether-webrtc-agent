import asyncio
from typing import Annotated
import re
import os
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatImage,
    FunctionContext
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
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

class DentalAssistantFunction(FunctionContext):
    @agents.llm.ai_callable(
        description="Called when user asked a Query that can be fetched using dental knowledge base for specific information"
    )
    async def query_dental_info(
        self,
        query: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user asked query to search in the dental knowledge base"
            )
        ],
    ):
        
        print(f"Answering from knowledgebase {query}")
        query_engine = index.as_query_engine(use_async=True)
        res = await query_engine.aquery(query)
        print("Query result:", res)
        return str(res)

    @agents.llm.ai_callable(
        description="Called when asked to evaluate dental issues using vision capabilities"
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
        description="Called when a user wants to book an appointment"
    )
    async def book_appointment(
        self,
        email: Annotated[str, agents.llm.TypeInfo(description="Email address")],
        name: Annotated[str, agents.llm.TypeInfo(description="Patient name")],
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

    @agents.llm.ai_callable(
        description="Assess the urgency of a dental issue"
    )
    async def assess_dental_urgency(
        self,
        symptoms: Annotated[str, agents.llm.TypeInfo(description="Dental symptoms")],
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
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Daela, a dental assistant for Knolabs Dental Agency. You are soft, caring with a bit of humour in you when responding. "
                    "You have access to a comprehensive dental knowledge base that includes various services, price info and different policies info that you can reference or query using tools available to provide accurate information about dental procedures, conditions, and care. "
                    "You offer appointment booking for dental care services, including urgent attention, routine check-ups, and long-term treatments available. An onsite appointment is required in most cases. "
                    "You can also analyze dental images to provide preliminary assessments, but always emphasize the need for professional in-person examination. "
                    "Provide friendly, professional assistance and emphasize the importance of regular dental care. "
                    "The users asking you questions could be of different age. so ask questions one by one. "
                    "Any query outside of the dental service & policies, politely reject stating your purpose."
                    "Keep your focus on try and get the patient's name and email address in sequence if not already provided while you help user. Encourage user to type email address to avoid any mistakes and reconfirm it after user provides it. "
                    "Remember every patient asking for information or help is a potential lead for the business so always try to get the patient info being polite and not pushing too much but using subtle discussion approach"
                    "If the care needed is not urgent, you can ask for image or ask user to show the dental area to use your vision capabilities to analyse the issue and offer assistance. "
                    "Always keep your conversation engaging, short and multiple interactions even when the information you are sharing is lengthy and try to offer the in-person appointment."
                ),
            )
        ]
    )

    latest_image: rtc.VideoFrame | None = None
    human_agent_present = False

    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
        chat_ctx=initial_ctx,
        fnc_ctx=DentalAssistantFunction(),
    )

    chat = rtc.ChatManager(ctx.room)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message and not human_agent_present:
            asyncio.create_task(assistant.say(msg.message, allow_interruptions=True))
        elif msg.message and human_agent_present and "help me" in msg.message.lower():
            asyncio.create_task(assistant.say(msg.message, allow_interruptions=True))

    @assistant.on("function_calls_finished")
    async def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        nonlocal human_agent_present
        if len(called_functions) == 0:
            return

        function = called_functions[0]
        function_name = function.call_info.function_info.name

        if function_name == "assess_dental_urgency":
            result = function.result
            if result == "call_human_agent":
                human_agent_phone = os.getenv('HUMAN_AGENT_PHONE')
                await create_sip_participant(human_agent_phone, ctx.room.name)
                human_agent_present = True
        elif function_name == "analyze_dental_image" and latest_image:
            user_instruction = function.call_info.arguments.get("user_msg")
            await assistant.say(user_instruction, allow_interruptions=True)

    assistant.start(ctx.room)
    await assistant.say(
        "Hello! I'm Daela, your dental assistant at Knolabs Dental Agency. "
        "Can I know if you are the patient or you're representing the patient?",
        allow_interruptions=True
    )

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)
        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            await asyncio.sleep(1)

async def create_sip_participant(phone_number, room_name):
    print("Trying to call an agent")
    livekit_api = api.LiveKitAPI(
        os.getenv('LIVEKIT_URL'),
        os.getenv('LIVEKIT_API_KEY'),
        os.getenv('LIVEKIT_API_SECRET')
    )

    await livekit_api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            sip_trunk_id=os.getenv('SIP_TRUNK_ID'),
            sip_call_to=phone_number,
            room_name=room_name,
            participant_identity=f"sip_{phone_number}",
            participant_name="Human Agent",
            play_ringtone=1
        )
    )
    await livekit_api.aclose()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))