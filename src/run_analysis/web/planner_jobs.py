"""Bounded ephemeral planner jobs; execution still uses the live pipeline/locks."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import monotonic
import logging


@dataclass
class _Job:
    key: str
    operation: str
    payload: dict
    updated: float
    state: str = 'queued'
    result: object = None
    error: dict | None = None


class PlannerJobs:
    def __init__(self,progress,*,capacity=128,max_pending=8,retention=1800,clock=monotonic):
        self.progress,self.capacity,self.max_pending,self.retention,self.clock=progress,capacity,max_pending,retention,clock
        self._jobs=OrderedDict();self._requests=OrderedDict();self._lock=Lock()
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='weekly-planner')
        self._closed=False

    def _prune(self):
        now=self.clock()
        for key,job in tuple(self._jobs.items()):
            if job.state not in ('queued','running') and now-job.updated>self.retention:
                del self._jobs[key]
        while len(self._jobs)>=self.capacity:
            key=next((key for key,job in self._jobs.items() if job.state not in ('queued','running')),None)
            if key is None:break
            del self._jobs[key]
        for key,(_,_,owner) in tuple(self._requests.items()):
            if owner not in self._jobs:del self._requests[key]
        while len(self._requests)>self.capacity:
            self._requests.popitem(last=False)

    def start(self,key,operation,payload,build):
        """Retry IDs are idempotent. Only adjacent read-refresh jobs coalesce.

        Never merge health/availability edits, or join an older refresh across
        an intervening edit. The single worker preserves submitted edit order.
        No DB connection is captured from a request thread.
        """
        with self._lock:
            self._prune()
            if self._closed:raise RuntimeError('Planner jobs are shutting down')
            previous=self._requests.get(key)
            if previous:
                if previous[:2]!=(operation,payload):
                    raise ValueError('Planner request ID already belongs to a different operation')
                return self._summary(self._jobs[previous[2]])
            active=[job for job in self._jobs.values() if job.state in ('queued','running')]
            if operation=='refresh' and active and active[-1].operation=='refresh':
                job=active[-1]
            else:
                if len(active)>=self.max_pending:raise RuntimeError('Planner queue is full; wait for the current update')
                job=_Job(key,operation,dict(payload),self.clock())
                self._jobs[key]=job
                self._executor.submit(self._run,job,build)
            self._requests[key]=(operation,dict(payload),job.key)
            return self._summary(job)

    def _run(self,job,build):
        with self._lock:
            if self._closed and job.state=='failed':return
            job.state='running';job.updated=self.clock()
        try:
            with self.progress.provide(),self.progress.observe(job.key):
                from ..planner_observation import finished
                result=build()
                finished()
            with self._lock:
                job.result=result;job.state='complete';job.updated=self.clock()
        except Exception as error:
            from fastapi import HTTPException
            from ..adaptive.live_bridge import LivePlannerReviewRequired
            if isinstance(error,HTTPException):status,detail=error.status_code,error.detail
            elif isinstance(error,LivePlannerReviewRequired):status,detail=409,str(error)
            else:
                status,detail=500,'Could not update your plan. Try again.'
                logging.getLogger(__name__).exception('Background planner job failed')
            with self._lock:
                job.error=dict(status=status,detail=detail);job.state='failed';job.updated=self.clock()

    @staticmethod
    def _summary(job):
        result=dict(job_id=job.key,state=job.state)
        if job.state=='complete':result['result']=job.result
        if job.state=='failed':result['error']=job.error
        return result

    def snapshot(self,key):
        with self._lock:
            self._prune()
            job=self._jobs.get(key)
            return self._summary(job) if job else None

    def resolve(self,key):
        with self._lock:
            return self._requests.get(key,(None,None,key))[2]

    def close(self):
        with self._lock:
            self._closed=True
            for job in self._jobs.values():
                if job.state=='queued':
                    job.state='failed';job.error=dict(status=503,detail='The app restarted; update the plan again.')
        self._executor.shutdown(wait=False,cancel_futures=True)
