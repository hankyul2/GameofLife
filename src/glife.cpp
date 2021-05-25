/****************************************/
/*                                      */
/*                                      */
/*  Game of Life with Pthread and CUDA  */
/*                                      */
/*  CSE561/SCE412 Project #2            */
/*  @ Ajou University                   */
/*                                      */
/*                                      */
/****************************************/

#include "glife.h"
#define MAX_NP 1000
using namespace std;

int gameOfLife(int argc, char *argv[]);
void singleThread(int, int, int);
void* workerThread(void *);
int nprocs;
GameOfLifeGrid* g_GameOfLifeGrid;

void make_index(int, int, int);
int dx [3] = {-1, 0, 1};
int dy [3] = {-1, 0, 1};

struct elem_idx {
    int s_r;
    int s_c;
    int e_r;
    int e_c;
};
elem_idx idxs[MAX_NP];

uint64_t dtime_usec(uint64_t start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

GameOfLifeGrid::GameOfLifeGrid(int rows, int cols, int gen)
{
  m_Generations = gen;
  m_Rows = rows;
  m_Cols = cols;

  m_Grid = (int**)malloc(sizeof(int*) * rows);
  if (m_Grid == NULL) 
    cout << "1 Memory allocation error " << endl;

  m_Temp = (int**)malloc(sizeof(int*) * rows);
  if (m_Temp == NULL) 
    cout << "2 Memory allocation error " << endl;

  m_Grid[0] = (int*)malloc(sizeof(int) * (cols*rows));
  if (m_Grid[0] == NULL) 
    cout << "3 Memory allocation error " << endl;

  m_Temp[0] = (int*)malloc(sizeof(int) * (cols*rows));	
  if (m_Temp[0] == NULL) 
    cout << "4 Memory allocation error " << endl;

  for (int i = 1; i < rows; i++) {
    m_Grid[i] = m_Grid[i-1] + cols;
    m_Temp[i] = m_Temp[i-1] + cols;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m_Grid[i][j] = m_Temp[i][j] = 0;
    }
  }
}

// Entry point
int main(int argc, char* argv[])
{
  if (argc != 7) {
    cout <<"Usage: " << argv[0] << " <input file> <display> <nprocs>"
           " <# of generation> <width> <height>" << endl;
    cout <<"\n\tnprocs = 0: Enable GPU" << endl;
    cout <<"\tnprocs > 0: Run on CPU" << endl;
    cout <<"\tdisplay = 1: Dump results" << endl;
    return 1;
  }

  return gameOfLife(argc, argv);
}

int gameOfLife(int argc, char* argv[])
{
  int cols, rows, gen;
  ifstream inputFile;
  int input_row, input_col, display;
  uint64_t difft, cuda_difft;
  pthread_t *threadID;

  inputFile.open(argv[1], ifstream::in);

  if (inputFile.is_open() == false) {
    cout << "The "<< argv[1] << " file can not be opend" << endl;
    return 1;
  }

  display = atoi(argv[2]);
  nprocs = atoi(argv[3]);
  gen = atoi(argv[4]);
  cols = atoi(argv[5]);
  rows = atoi(argv[6]);

  g_GameOfLifeGrid = new GameOfLifeGrid(rows, cols, gen);

  while (inputFile.good()) {
    inputFile >> input_row >> input_col;
    if (input_row >= rows || input_col >= cols) {
      cout << "Invalid grid number" << endl;
      return 1;
    } else
      g_GameOfLifeGrid->setCell(input_row, input_col);
  }

  // Start measuring execution time
  difft = dtime_usec(0);

  // TODO: YOU NEED TO IMPLMENT THE SINGLE THREAD, PTHREAD, AND CUDA
  if (nprocs == 0) {
    // Running on GPU
    // cuda_difft = runCUDA(rows, cols, gen, g_GameOfLifeGrid, display);
    cuda_difft = 1;
  } else if (nprocs == 1) {
    // Running a single thread
    singleThread(rows, cols, gen);
  } else { 
    // Running multiple threads (pthread)
    
    make_index(rows, cols, nprocs);
    
    int status;
    pthread_barrier_t barrier;
    pthread_t thread_group[nprocs];
    pthread_barrier_init(&barrier, NULL, nprocs);

    myargs thread_vars[nprocs];
    for (int i=0; i<nprocs; i++){
      thread_vars[i] = { rows, cols, gen, i, &barrier};
    }
    
    for(int i=0; i<nprocs; i++){
      pthread_create(&thread_group[i], NULL, workerThread, (void*)&thread_vars[i]);
    }
    // cout << "hello" << endl;
    for (int i=0; i<nprocs; i++){
      pthread_join(thread_group[i], NULL);
    }
  }

  difft = dtime_usec(difft);

  // Print indices only for running on CPU(host).
  if (display && nprocs) {
    g_GameOfLifeGrid->dump();
    g_GameOfLifeGrid->dumpIndex();
  }

  if (nprocs) {
    // Single or multi-thread execution time 
    cout << "Execution time(seconds): " << difft/(float)USECPSEC << endl;
  } else {
    // CUDA execution time
    cout << "Execution time(seconds): " << cuda_difft/(float)USECPSEC << endl;
  }
  inputFile.close();
  
  return 0;
}

// TODO: YOU NEED TO IMPLMENT SINGLE THREAD
void singleThread(int rows, int cols, int gen)
{
  while(g_GameOfLifeGrid->decGen()){
    g_GameOfLifeGrid->next();
  }
}

// TODO: YOU NEED TO IMPLMENT PTHREAD
void *workerThread(void *arg)
{
  myargs *args = (myargs *) arg;
  int n = args->gens;
  // while(g_GameOfLifeGrid->decGen()){
  while(n--){
    g_GameOfLifeGrid->next(args);
  }
}

// HINT: YOU MAY NEED TO FILL OUT BELOW FUNCTIONS OR CREATE NEW FUNCTIONS
void GameOfLifeGrid::next(myargs * args)
{
  elem_idx idx = idxs[args->t_id];
  int s_r = idx.s_r, s_c = idx.s_c, e_r = idx.e_r, e_c = idx.e_c;
  int w = args->cols;
  while(s_r * w + s_c < e_r * w + e_c){
    if(canGoLive(s_r, s_c, getNumOfNeighbors(s_r, s_c))) live(s_r, s_c);
    else dead(s_r, s_c);
    if((++s_c)==w) {s_c=0; s_r++;}
  }
  cout.flush();
  pthread_barrier_wait(args->barrier);
  s_r = idx.s_r, s_c = idx.s_c, e_r = idx.e_r, e_c = idx.e_c;
  while(s_r * w + s_c < e_r * w + e_c){
    updateGrid(s_r, s_c);
    if((++s_c)==w) {s_c=0; s_r++;}
  }
  pthread_barrier_wait(args->barrier);
}

void GameOfLifeGrid::next()
{
  for(int r=0; r<m_Rows; r++){
    for(int c=0; c<m_Cols; c++){
      if(canGoLive(r, c, getNumOfNeighbors(r, c))) live(r, c);
      else dead(r, c);
    }
  }
  for(int r=0; r<m_Rows; r++){
    for(int c=0; c<m_Cols; c++){
      updateGrid(r, c);
    }
  }
}

bool GameOfLifeGrid::canGoLive(int row, int col, int N)
{
  if(isLive(row, col) and (N == 2 or N==3)) return true;
  if(!isLive(row, col) and N==3) return true;
  return false;
}

bool GameOfLifeGrid::isOoB(int row, int col)
{
  if(row < 0 or row >= m_Rows) return true;
  if(col < 0 or col >= m_Cols) return true;
  return false;
}


// TODO: YOU MAY NEED TO IMPLMENT IT TO GET NUMBER OF NEIGHBORS 
int GameOfLifeGrid::getNumOfNeighbors(int row, int col)
{
  int new_x, new_y, numOfNeighbors=0;
  for(auto x : dx){
    for(auto y: dy){
      new_x = row + x; new_y = col + y;
      (x==0 and y==0) or isOoB(new_x, new_y) or !isLive(new_x, new_y) ? 0 : numOfNeighbors++;
    }
  }
  return numOfNeighbors;
}

void GameOfLifeGrid::dump() 
{
  cout << "===============================" << endl;

  for (int i = 0; i < m_Rows; i++) {
    cout << "[" << i << "] ";
    for (int j = 0; j < m_Cols; j++) {
      if (m_Grid[i][j] == 1)
        cout << "*";
      else
        cout << "o";
    }
    cout << endl;
  }
  cout << "===============================\n" << endl;
}

void GameOfLifeGrid::dumpIndex()
{
  int ans = 0;
  cout << ":: Dump Row Column indices" << endl;
  for (int i=0; i < m_Rows; i++) {
    for (int j=0; j < m_Cols; j++) {
      if (m_Grid[i][j]) { cout << i << " " << j << endl; ans++; }
    }
  }
  cout << ans << endl;
}

void make_index(int w, int h, int np){
    int count = w * h / np;
    int rem = w * h % np;
    int mv_sum = 0;
    for(int i=0; i<np; i++){
        if (i==np-1 and rem != 0) {
            count += rem;
        }
        int s_r = mv_sum / w, s_c = mv_sum % w; mv_sum += count;
        int e_r = mv_sum / w, e_c = mv_sum % w;
        idxs[i] = {s_r, s_c, e_r, e_c};
    }
}