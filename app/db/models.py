from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from app.db.session import Base
import uuid

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    phone = Column(String(20), nullable=True)
    role = Column(String, default="USER")
    grade_level = Column(Integer, nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    avatar_file_id = Column(UUID(as_uuid=True), unique=True, nullable=True)
    avatar_url = Column(Text, nullable=True)
    bio = Column(Text, nullable=True)
    address_id = Column(String, unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class Address(Base):
    __tablename__ = "addresses"
    id = Column(String, primary_key=True)
    detail = Column(String, nullable=False)
    ward = Column(String, nullable=False)
    district = Column(String, nullable=True)
    city = Column(String, nullable=False)
    country = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class UserQuota(Base):
    __tablename__ = "user_quotas"
    user_id = Column(String, primary_key=True)
    total_credits = Column(Integer, default=0)
    used_credits = Column(Integer, default=0)
    remaining_credits = Column(Integer, default=0)
    total_chat_messages = Column(Integer, default=0)
    used_chat_messages = Column(Integer, default=0)
    remaining_chat_messages = Column(Integer, default=0)
    total_create_videos = Column(Integer, default=0)
    used_create_videos = Column(Integer, default=0)
    remaining_create_videos = Column(Integer, default=0)
    total_flashcards = Column(Integer, default=0)
    used_flashcards = Column(Integer, default=0)
    remaining_flashcards = Column(Integer, default=0)
    total_voice_calls = Column(Integer, default=0)
    used_voice_calls = Column(Integer, default=0)
    remaining_voice_calls = Column(Integer, default=0)
    last_reconciled_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class QuotaLedger(Base):
    __tablename__ = "quota_ledgers"
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    action_type = Column(String)
    credit_delta = Column(Integer, default=0)
    chat_message_delta = Column(Integer, default=0)
    create_video_delta = Column(Integer, default=0)
    flashcard_delta = Column(Integer, default=0)
    voice_call_delta = Column(Integer, default=0)
    reference_type = Column(String, nullable=True)
    reference_id = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

class FlashcardSet(Base):
    __tablename__ = "flashcard_sets"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True, index=True)
    title = Column(String(255))
    description = Column(String(500), nullable=True)
    subject_id = Column(String, nullable=True)
    grade_level = Column(Integer, nullable=True)
    ai_model = Column(String, nullable=True)
    generation_status = Column(String, default="PENDING")
    is_of_system = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    view_count = Column(Integer, default=0)
    clone_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    do_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class Flashcard(Base):
    __tablename__ = "flashcards"
    id = Column(String, primary_key=True)
    flashcard_set_id = Column(String, index=True)
    front_image_file_id = Column(UUID(as_uuid=True), nullable=True)
    back_image_file_id = Column(UUID(as_uuid=True), nullable=True)
    audio_file_id = Column(UUID(as_uuid=True), nullable=True)
    type = Column(String, default="BASIC")
    front_content = Column(Text)
    back_content = Column(Text)
    front_image_url = Column(Text, nullable=True)
    back_image_url = Column(Text, nullable=True)
    audio_url = Column(Text, nullable=True)
    card_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())

class Video(Base):
    __tablename__ = "videos"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True, index=True)
    video_file_id = Column(UUID(as_uuid=True), nullable=True)
    thumbnail_file_id = Column(UUID(as_uuid=True), nullable=True)
    title = Column(String(255))
    description = Column(String(500), nullable=True)
    subject = Column(String(100), nullable=True)
    grade_level = Column(Integer, nullable=True)
    ai_model = Column(String, nullable=True)
    generation_status = Column(String, default="PENDING")
    video_url = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    script_content = Column(Text, nullable=True)
    transcript = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class Quizz(Base):
    __tablename__ = "quizzes"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True, index=True)
    is_of_system = Column(Boolean, default=True)
    title = Column(String(255))
    description = Column(String(500), nullable=True)
    subject_id = Column(String, nullable=True)
    test_category_id = Column(String, nullable=True)
    grade_level = Column(Integer, nullable=True)
    ai_model = Column(String, nullable=True)
    generation_status = Column(String, nullable=True)
    time_limit = Column(Integer, nullable=True)
    passing_score_percentage = Column(Integer, nullable=True)
    points_to_complete = Column(Integer, default=0)
    points_to_earn = Column(Integer, default=0)
    gift_credits = Column(Integer, default=0)
    max_attempts_per_user = Column(Integer, default=0)
    is_public = Column(Boolean, default=False)
    is_featured = Column(Boolean, default=False)
    total_attempts = Column(Integer, default=0)
    total_completions = Column(Integer, default=0)
    average_score = Column(Numeric(5, 2), default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class QuizQuestion(Base):
    __tablename__ = "quiz_questions"
    id = Column(String, primary_key=True)
    quiz_id = Column(String, index=True)
    question_image_file_id = Column(UUID(as_uuid=True), nullable=True)
    question_title = Column(String(255))
    question_text = Column(Text)
    question_image_url = Column(Text, nullable=True)
    correct_answer = Column(ARRAY(String))
    explanation = Column(Text, nullable=True)
    question_type = Column(String)
    points = Column(Integer, default=1)
    question_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())

class QuizAnswerOption(Base):
    __tablename__ = "quiz_answer_options"
    id = Column(String, primary_key=True)
    question_id = Column(String, index=True)
    option_image_file_id = Column(UUID(as_uuid=True), nullable=True)
    option_label = Column(String)
    option_text = Column(Text)
    option_image_url = Column(Text, nullable=True)
    option_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())

class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    title = Column(String(255), nullable=True)
    subject = Column(String(100), nullable=True)
    grade_level = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    message_id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    attachment_file_id = Column(UUID(as_uuid=True), nullable=True)
    role = Column(String)
    content_type = Column(String)
    content = Column(Text, default="")
    has_attachment = Column(Boolean, default=False)
    attachment_type = Column(String, nullable=True)
    attachment_url = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

class Subject(Base):
    __tablename__ = "subjects"
    subject_id = Column(String, primary_key=True)
    icon_name = Column(String, nullable=True)
    subject_name = Column(String(100))
    subject_code = Column(String(20))
    description = Column(Text, nullable=True)
    color_hex = Column(String, nullable=True)
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class TestCategory(Base):
    __tablename__ = "test_categories"
    category_id = Column(String, primary_key=True)
    icon_name = Column(String, nullable=True)
    category_name = Column(String(100))
    category_code = Column(String(20))
    description = Column(Text, nullable=True)
    display_order = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

class File(Base):
    __tablename__ = "files"
    id = Column(UUID(as_uuid=True), primary_key=True)
    owner_id = Column(String, nullable=True, index=True)
    bucket_name = Column(String)
    storage_key = Column(String, unique=True, index=True)
    original_file_name = Column(Text)
    stored_file_name = Column(Text)
    file_extension = Column(String(20))
    file_type = Column(String)
    mime_type = Column(String)
    size_bytes = Column(Integer)
    storage_url = Column(Text)
    cdn_url = Column(Text, nullable=True)
    signed_url = Column(Text, nullable=True)
    signed_url_expires_at = Column(DateTime(timezone=True), nullable=True)
    visibility = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    deleted_by_user_id = Column(String, nullable=True)
    delete_reason = Column(Text, nullable=True)

class Song(Base):
    __tablename__ = "songs"
    id = Column(String(64), primary_key=True, index=True)
    title = Column(String, nullable=False)
    artists = Column(ARRAY(String), nullable=True)
    genre = Column(String, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    has_reference_audio = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class SongScore(Base):
    __tablename__ = "song_scores"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    song_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    upload_file_path = Column(String, nullable=False)

    total_score = Column(Float, nullable=False)
    grade = Column(String, nullable=False)
    pitch_score = Column(Float, nullable=False)
    rhythm_score = Column(Float, nullable=False)
    stability_score = Column(Float, nullable=False)
    dynamics_score = Column(Float, nullable=False)

    processing_time_ms = Column(Float, nullable=False)
    raw_result_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
