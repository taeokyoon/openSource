import express from 'express';

const app = express();

// 루트("/") 요청에 응답
app.get('/', (req, res) => {
  res.send('<h2>Hi there!</h2>');
});

// 3000 포트에서 서버 실행
app.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});
