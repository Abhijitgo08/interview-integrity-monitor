from django.db import models
from django.utils import timezone

class Candidate(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    resume = models.FileField(upload_to='resumes/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class InterviewSession(models.Model):
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    # State tracking for violations
    last_face_seen = models.DateTimeField(auto_now_add=True)
    last_audio_activity = models.DateTimeField(auto_now_add=True)

    # Final scoring fields (populated at end)
    final_score = models.FloatField(default=100.0)
    violation_count = models.IntegerField(default=0)
    risk_level = models.CharField(max_length=20, default='Green') # Green, Yellow, Red

    def __str__(self):
        return f"Session {self.id} - {self.candidate.name}"

class ViolationLog(models.Model):
    VIOLATION_TYPES = [
        ('FACE_MISSING', 'Face Missing'),
        ('MULTIPLE_FACES', 'Multiple Faces'),
        ('FACE_ORIENTATION', 'Poor Face Orientation'),
        ('AUDIO_SILENCE', 'Prolonged Silence'),
        ('TAB_SWITCH', 'Tab Switch / Window Blur'),
    ]
    
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name='violations')
    violation_type = models.CharField(max_length=50, choices=VIOLATION_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.TextField(blank=True, null=True) # e.g., "Silence for 15s"
    
    def __str__(self):
        return f"{self.violation_type} at {self.timestamp}"

class AudioActivityLog(models.Model):
    # Logs summary of audio activity (optional, primarily for debugging or detailed checking)
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    volume_level = models.FloatField() # 0.0 to 1.0 (or dB)

class TabSwitchLog(models.Model):
    # Specific detail log for tab switches if we want separate tracking
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=20) # 'blur', 'focus', 'hidden', 'visible'
