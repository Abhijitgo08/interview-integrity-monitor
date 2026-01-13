from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json
import base64
import numpy as np
import cv2
from .models import Candidate, InterviewSession, ViolationLog, AudioActivityLog, TabSwitchLog
from .ml_utils import FaceAnalyzer

# Config / Constants
FACE_ABSENCE_THRESHOLD_SECONDS = 5
SILENCE_THRESHOLD_SECONDS = 10
VIOLATION_PENALTIES = {
    'FACE_MISSING': 5,
    'MULTIPLE_FACES': 10,
    'FACE_ORIENTATION': 2,
    'AUDIO_SILENCE': 5,
    'TAB_SWITCH': 8,
}

def get_session(session_id):
    try:
        return InterviewSession.objects.get(id=session_id, is_active=True)
    except InterviewSession.DoesNotExist:
        return None

def index(request):
    return render(request, 'index.html')

def monitor(request):
    return render(request, 'monitor.html')

@csrf_exempt
def start_interview(request):
    if request.method == 'POST':
        try:
            # Handle multipart/form-data or JSON
            if request.content_type.startswith('multipart/form-data'):
                data = request.POST
                resume = request.FILES.get('resume')
            else:
                data = json.loads(request.body)
                resume = None
            
            email = data.get('email', 'unknown@example.com')
            name = data.get('name', 'Unknown Candidate')
            
            # Get or create candidate
            candidate, created = Candidate.objects.get_or_create(email=email, defaults={'name': name})
            
            # Update resume if provided
            if resume:
                candidate.resume = resume
                candidate.save()
            
            # Start session
            session = InterviewSession.objects.create(candidate=candidate)
            
            return JsonResponse({'status': 'success', 'session_id': session.id})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

def get_candidate_details(request):
    session_id = request.GET.get('session_id')
    session = get_session(session_id)
    if not session:
        return JsonResponse({'status': 'error', 'message': 'Session not found'}, status=404)
    
    candidate = session.candidate
    resume_url = candidate.resume.url if candidate.resume else None
    
    return JsonResponse({
        'status': 'success',
        'name': candidate.name,
        'email': candidate.email,
        'resume_url': resume_url
    })

@csrf_exempt
def end_interview(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        session = get_session(session_id)
        if not session:
             return JsonResponse({'status': 'error', 'message': 'Session not found'}, status=404)
             
        session.is_active = False
        session.end_time = timezone.now()
        
        # Calculate final score
        score = 100
        violations = session.violations.all()
        for v in violations:
            penalty = VIOLATION_PENALTIES.get(v.violation_type, 0)
            score -= penalty
        
        session.final_score = max(0, score)
        session.violation_count = violations.count()
        
        if score > 85:
            session.risk_level = 'Green'
        elif score > 50:
            session.risk_level = 'Yellow'
        else:
            session.risk_level = 'Red'
            
        session.save()
        
        return JsonResponse({'status': 'success', 'final_score': session.final_score, 'risk_level': session.risk_level})
    return JsonResponse({'status': 'error'}, status=405)

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            frame_data = data.get('frame') # base64
            
            session = get_session(session_id)
            if not session:
                return JsonResponse({'status': 'error', 'message': 'Invalid Session'}, status=404)

            # Decode image
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Analyze
            analyzer = FaceAnalyzer()
            analysis = analyzer.process_frame(img)
            
            status = analysis['status']
            violation = None
            
            now = timezone.now()
            
            if status == 'no_face':
                # Check duration
                time_since_last = (now - session.last_face_seen).total_seconds()
                if time_since_last > FACE_ABSENCE_THRESHOLD_SECONDS:
                    # Log violation only if we haven't logged one recently for this burst? 
                    # For simplicity, we log it. To avoid spam, we could check last violation time.
                    # But for now, we just log and maybe debounce on frontend or here.
                    # Let's debounce: if last violation of this type was < 2s ago, skip.
                    last_v = ViolationLog.objects.filter(session=session, violation_type='FACE_MISSING').order_by('-timestamp').first()
                    if not last_v or (now - last_v.timestamp).total_seconds() > 2:
                        violation = ViolationLog.objects.create(
                            session=session,
                            violation_type='FACE_MISSING',
                            details=f"Face missing for > {time_since_last:.1f}s"
                        )
            elif status == 'multiple_faces':
                session.last_face_seen = now # Technically face is there
                violation = ViolationLog.objects.create(
                    session=session,
                    violation_type='MULTIPLE_FACES',
                    details="Multiple faces detected"
                )
            else: # OK
                session.last_face_seen = now
            
            session.save()
            
            return JsonResponse({
                'status': 'success', 
                'face_status': status,
                'violation': violation.violation_type if violation else None
            })
            
        except Exception as e:
            # print(e) # Debug
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error'}, status=405)

@csrf_exempt
def log_tab_switch(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        event_type = data.get('event_type') # 'blur' or 'hidden'
        
        session = get_session(session_id)
        if session:
             ViolationLog.objects.create(
                 session=session,
                 violation_type='TAB_SWITCH',
                 details=f"Page {event_type}"
             )
             return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'}, status=400)

@csrf_exempt
def send_audio_activity(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        session_id = data.get('session_id')
        is_silent = data.get('is_silent') # boolean
        
        session = get_session(session_id)
        if session:
            now = timezone.now()
            if not is_silent:
                session.last_audio_activity = now
                session.save()
            else:
                # Check silence duration
                time_since_active = (now - session.last_audio_activity).total_seconds()
                if time_since_active > SILENCE_THRESHOLD_SECONDS:
                     # Debounce
                    last_v = ViolationLog.objects.filter(session=session, violation_type='AUDIO_SILENCE').order_by('-timestamp').first()
                    if not last_v or (now - last_v.timestamp).total_seconds() > 5:
                        ViolationLog.objects.create(
                            session=session,
                            violation_type='AUDIO_SILENCE',
                            details=f"Silence for > {time_since_active:.1f}s"
                        )
                        return JsonResponse({'status': 'success', 'violation': 'AUDIO_SILENCE'})
            return JsonResponse({'status': 'success', 'violation': None})
    return JsonResponse({'status': 'error'}, status=400)
