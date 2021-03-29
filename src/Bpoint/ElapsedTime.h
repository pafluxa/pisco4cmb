/** @file ElapsedTime.h

 Uses the linux monotonic timer to calculate elapsed time (1 ns resolution)
 Link with -lrt

 @par CREATION
 @author Michael K. Brewer
 @date   04.30.2015
*/

#ifndef INCElapsedTimeh
#define INCElapsedTimeh

#include <time.h>
/** Structure containing monotonic timer count and elapsed time */
struct ElapsedTimeStruct {
/** monotonic timer count */
  timespec curtime;
/** elapsed time in seconds */
  double etime;
};

typedef struct ElapsedTimeStruct ElapsedTime_t;

/** Class containing elapsed time methods.
*   Uses the linux monotonic timer to calculate elapsed time (1 ns resolution)
*/
class ElapsedTime
{

public:
/** calculate elapsed time
  * @param [in] btime: ElapsedTime_t struct containing beginning HPET timer count 
  * @retval ElapsedTime_t struct containing current monotonic timer count and time elapsed since btime.curtime 
*/
  ElapsedTime_t elapse(ElapsedTime_t btime);
};

#endif

