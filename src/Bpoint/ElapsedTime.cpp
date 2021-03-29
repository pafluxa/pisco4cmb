/** @file ElapsedTime.cpp

 
 @par CREATION
 @author Michael K. Brewer
 @date   04.30.2015
*/

#include "ElapsedTime.h"

ElapsedTime_t ElapsedTime::elapse(ElapsedTime_t btime)
{


  ElapsedTime_t mytime;
  timespec etime;
  const int ONESEC = 1000000000;

  clock_gettime(CLOCK_MONOTONIC, &mytime.curtime); 

  etime.tv_sec = mytime.curtime.tv_sec - btime.curtime.tv_sec;
  etime.tv_nsec =  mytime.curtime.tv_nsec - btime.curtime.tv_nsec;
  if (etime.tv_nsec < 0) {
    etime.tv_sec--;
    etime.tv_nsec += ONESEC;
  }
	  
  mytime.etime  = (double) etime.tv_sec;
  mytime.etime += (double) etime.tv_nsec / ONESEC;

  return(mytime);
}



